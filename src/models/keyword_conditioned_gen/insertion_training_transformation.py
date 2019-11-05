from typing import Callable, List, Generator, Iterable, Union, Tuple
import torch
import torch.nn
import itertools
import random
import logging
from collections import defaultdict

_DECODER_INPUT = List[torch.LongTensor]
_NUMBER_ARRAY = torch.LongTensor
_PRED_LOC = _NUMBER_ARRAY # a batch of indices to slots
_PRED_TOK = _NUMBER_ARRAY # a batch of indices to vocabulary set
_PRED_WEIGHT = torch.Tensor # a batch of indices to vocabulary set

INSERTION_ORDER = Generator[Tuple[_DECODER_INPUT, _PRED_LOC, _PRED_TOK, _PRED_WEIGHT], None, None]

class TrainingTransformation:
    """
    A general interface that specifies how an insertion transformer should be trained.
    """
    def __call__(self, *args, **kwargs):
        yield from self.example_gen(*args, **kwargs)

    def example_gen(self, *args, **kwargs) -> INSERTION_ORDER:
        raise NotImplemented

class UniformInsTrans(TrainingTransformation):
    def __init__(self, eos_id, use_slot_loss=True):
        self.eos_id = eos_id
        self._use_slot_loss = use_slot_loss

    END_OF_SPAN_TOKEN = "@End-of-Span@"

    def example_gen(self, tkwds: List[torch.LongTensor], tgts: List[torch.LongTensor]):
        batch_dec_inps, batch_target_slots, batch_target_toks, batch_clweights = [], [], [], []

        for tkwd, tgt in zip(tkwds, tgts):
            dec_inp, target_slots, target_toks, target_weights = self._process_single_example(tgt, tkwd)
            batch_dec_inps.append(dec_inp)
            batch_target_slots.append(target_slots)
            batch_target_toks.append(target_toks)
            batch_clweights.append(target_weights)

        yield batch_dec_inps, batch_target_slots, batch_target_toks, batch_clweights

    def _process_single_example(self, tgt, tkwd):
        # 1. sample a K for auxiliary words, from zero to all
        tgt_len = tgt.size()[0]
        tkwd_len = tkwd.size()[0]
        k = random.randint(0, tgt_len - tkwd_len + 1)

        # 2. based on given keywords, and the sampled k,
        # find the selected locations of targets, and consequently the not selected locations
        selected_loc, complement_loc = self.get_sel_compl_locs(tgt, tkwd, k)

        output = self._get_inp_tgt_from_locations(tgt, selected_loc, complement_loc)
        dec_inp, target_slots, target_toks, target_weights = output

        return dec_inp, target_slots, target_toks, target_weights

    def _get_inp_tgt_from_locations(self, tgt: torch.Tensor, selected_loc: List[int], complement_loc: List[int]):
        tgt_len = tgt.size()[0]
        dec_inp = tgt.gather(dim=0, index=tgt.new_tensor(selected_loc))
        if len(complement_loc) > 0:
            # determine the corresponding slot for each remaining word
            target_toks = tgt.gather(dim=0, index=tgt.new_tensor(complement_loc))
            # slots is considered the weights for the same
            slots = []
            for i, x in enumerate(complement_loc):
                if i == 0:
                    slots.append(x)
                else:
                    prev_x = complement_loc[i - 1]
                    prev_slot = slots[-1]
                    slots.append(x - prev_x - 1 + prev_slot)

            if self._use_slot_loss:
                # Slot-level loss requires every empty slot to predict the <EoSpan> token.
                # Thus extend slots and tokens for these empty locations.
                # Weights will be computed automatically from the slots.
                empty_slots = sorted(set(range(len(selected_loc) + 1)) - set(slots))
                slots.extend(empty_slots)
                empty_toks = tgt.new_tensor([self.eos_id] * len(empty_slots))
                target_toks = torch.cat([target_toks, empty_toks], dim=-1)

            target_slots = tgt.new_tensor(slots)
            target_weights = tgt.new_tensor(self.get_weights_for_slots(slots), dtype=torch.float)

        else:
            # For both slot- and seq-level loss, all slots must predict <EoSpan> token if full sequence is given.
            target_slots = torch.arange(tgt_len + 1, dtype=tgt.dtype, device=tgt.device)
            target_toks = torch.full_like(target_slots, self.eos_id)
            target_weights = torch.ones_like(target_slots).float()

        return dec_inp, target_slots, target_toks, target_weights

    def get_weights_for_slots(self, slots: List[int]) -> List[float]:
        # Uniform weights treat each
        slot_counts = dict((k, len(list(g)))for k, g in itertools.groupby(slots))
        weights = [1./slot_counts[s] for s in slots]    # uniform weights for every slot
        return weights

    def get_sel_compl_locs(self, tgt: torch.LongTensor, tkwd: torch.LongTensor, selected_num: int):
        tgt_len = tgt.size()[0]
        tkwd_len = tkwd.size()[0]

        # 2. find keyword locations
        tgt_arr, tkwd_arr = tgt.tolist(), tkwd.tolist()
        tkwd_loc, start = [], 0
        for k in tkwd_arr:
            subarr = tgt_arr[start:]
            if k not in subarr:
                continue

            loc = start + subarr.index(k)
            tkwd_loc.append(loc)
            start = loc + 1

        # 3. and sample auxiliary words other than keywords
        remaining_loc = list(set(range(tgt_len)) - set(tkwd_loc))
        random.shuffle(remaining_loc)
        aux_loc = remaining_loc[:selected_num]
        selected_loc = sorted(aux_loc + tkwd_loc)

        # 4. return the complement words for keywords + aux. words
        complement_loc = sorted(set(range(tgt_len)) - set(selected_loc))

        return selected_loc, complement_loc

class BSTInsTrans(UniformInsTrans):

    def __init__(self, eos_id, use_slot_loss=False, tao=1., reverse=False):
        super(BSTInsTrans, self).__init__(eos_id, use_slot_loss)
        self._tao = tao
        self._reverse = reverse

    def get_weights_for_slots(self, slots: List[int]) -> List[float]:
        import math
        # assume the slots had been arranged that the same keys were put together.
        counts = [(k, len(list(g)))for k, g in itertools.groupby(slots)]
        def _get_weights_from_count(count: int) -> List[float]:
            center = (count - 1.) / 2.
            if self._reverse:
                weights = [math.exp(abs(center - i) / self._tao) for i in range(count)]
            else:
                weights = [math.exp(- abs(center - i) / self._tao) for i in range(count)]

            total = sum(weights)
            return [w / total for w in weights]

        output = []
        for _, count in counts:
            output.extend(_get_weights_from_count(count))

        return output

class DecoupledInsTrans(TrainingTransformation):

    END_OF_SPAN_TOKEN = "<eospan>"
    MASK_TOKEN = "<mask>"

    logger = logging.getLogger(__name__)

    def __init__(self, mask_id):
        self.mask_id = mask_id

    def example_gen(self, tkwds: List[torch.LongTensor], tgts: List[torch.LongTensor], skill=1):
        batch_slot_data, batch_content_data, batch_dual_data = [], [], []

        for tkwd, tgt in zip(tkwds, tgts):
            training_data = self._process_single_example(tgt, tkwd, skill)
            slot_dec_data, cont_dec_data, dual_data = training_data
            batch_slot_data.append(slot_dec_data)
            batch_content_data.append(cont_dec_data)
            batch_dual_data.append(dual_data)

        yield batch_slot_data, batch_content_data, batch_dual_data

    def _process_single_example(self, tgt, tkwd, skill):
        """
        Use an illustrative example for the function.

                                   0   1   2   3   4   5   6   7   8
        Given a sentence target: [w1, w2, k1, w3, w4, w5, k2, k3, w6]
        keyword locations: tkwd_loc = [2, 6, 7]

        suppose k = 2, select two additional words [w2, w6] despite the keywords
        selected locations: selected_loc = [1, 2, 6, 7, 8]
        unselected locations: complement_loc = [0, 3, 4, 5]
        5 selected words, thus 6 slots [0, 6) in total.
        slots of all complement words: complement_slot = [0, 2, 2, 2]

        1. for slot decoder model
          input:           slot_dec_inp = [  w2, k1, k2, k3, w6  ]
         target:        slot_dec_target = [ 1,  0,  1,  0,  0,  0]
        weights: slot_dec_target_weight = [ 2,  1,  4,  1,  1,  1]

        2. for content decoder model
            index: [     0,  1,  2,      3,  4,  5,  6]
            input: [<mask>, w2, k1, <mask>, k2, k3, w6]
           target_loc: cont_dec_target_loc    = [ 0,  3,  3,  3,  1,  2,  4,  5,  6]
          target_word: cont_dec_target_word   = [w1, w3, w4, w5, w2, k1, k2, k3, w6]
        target_weight: cont_dec_target_weight = [ 1, .3, .3, .3,  1,  1,  1,  1,  1]

        3. for dual model
        input:  [w2, k1, k2, k3, w6]
        target: [ 1,  0,  0,  0,  1]
        """
        # 1. sample a K for auxiliary words, from zero to all
        tgt_len = tgt.size()[0]
        tkwd_len = tkwd.size()[0]
        # skill is clipped to [0, 1].
        # as long as the skill increases, the number of missing words also grows.
        skill = max(0, min(skill, 1))
        selected_ratio = 1 - skill
        k = random.randint(int((tgt_len - tkwd_len) * selected_ratio), (tgt_len - tkwd_len))

        # 2. based on given keywords, and the sampled k,
        # find the selected locations of targets, and consequently the not selected locations
        tkwd_loc, selected_loc, complement_loc = self.get_sel_compl_locs(tgt, tkwd, k)

        self.logger.debug(f"========== processing a single example ===========\n"
                          f"skill={skill} k={k} tgt_len={tgt_len} tkwd_len={tkwd_len}\n"
                          f"sentence={tgt.tolist()}\n"
                          f"keywords={tkwd.tolist()}\n"
                          f"kwd_loc={tkwd_loc}\n"
                          f"selected={selected_loc}\n"
                          f"complement_loc={complement_loc}")

        if len(complement_loc) > 0:
            # ----------
            # build target for slot decoder
            slot_num = len(selected_loc) + 1
            slot_dec_inp, complement_slots = self._get_slot_dec_data(tgt, selected_loc, complement_loc)
            slot_dec_target = list(1 if i in complement_slots else 0 for i in range(slot_num))

            slot_counts = defaultdict(lambda: 0, ((k, len(list(g))) for k, g in itertools.groupby(complement_slots)))
            slot_dec_target_weight = list(slot_counts[i] + 1 for i in range(slot_num))

            self.logger.debug(f"---------- slot decoder data ------------\n"
                              f"inputs={slot_dec_inp} length={len(slot_dec_inp)}\n"
                              f"complement_slots={complement_slots}\n"
                              f"targets={slot_dec_target}\n"
                              # f"target_weights={slot_dec_target_weight}" weights are forbidden
                              )

            # ----------
            # build target for content decoder
            output = self._get_content_dec_data(tgt, slot_dec_inp, complement_slots, complement_loc)
            cont_dec_inp, cont_dec_target_loc, cont_dec_target_word = output
            cont_dec_target_weight = self.get_weights_for_slots(cont_dec_target_loc)

            slot_dec_inp = tgt.new_tensor(slot_dec_inp, dtype=torch.long)
            slot_dec_target = tgt.new_tensor(slot_dec_target, dtype=torch.long)
            slot_dec_target_weight = tgt.new_tensor(slot_dec_target_weight, dtype=torch.float32)

            cont_dec_inp = tgt.new_tensor(cont_dec_inp)
            cont_dec_target_loc = tgt.new_tensor(cont_dec_target_loc)
            cont_dec_target_word = tgt.new_tensor(cont_dec_target_word)
            cont_dec_target_weight = tgt.new_tensor(cont_dec_target_weight, dtype=torch.float32)

        else:
            slot_dec_inp = tgt.gather(dim=0, index=tgt.new_tensor(selected_loc))
            slot_dec_target = tgt.new_tensor([0] * (tgt_len + 1), dtype=torch.long)
            slot_dec_target_weight = tgt.new_tensor([1] * (tgt_len + 1), dtype=torch.float32)

            # no word need to be inserted, content decoder is trained with zero weights
            cont_dec_inp = slot_dec_inp
            cont_dec_target_loc = tgt.new_zeros((1,), dtype=torch.long)
            cont_dec_target_word = tgt.new_zeros((1,), dtype=torch.long)
            cont_dec_target_weight = tgt.new_zeros((1,), dtype=torch.float32)

        # ----------
        # build target for dual model inp
        dual_inp = slot_dec_inp
        dual_target = tgt.new_tensor(self._get_removal_model_data(tkwd_loc, selected_loc))

        slot_dec_data = (slot_dec_inp, slot_dec_target, slot_dec_target_weight)
        cont_dec_data = (cont_dec_inp, cont_dec_target_loc, cont_dec_target_word, cont_dec_target_weight)
        dual_data = (dual_inp, dual_target)

        return slot_dec_data, cont_dec_data, dual_data

    def _get_removal_model_data(self, tkwd_loc, selected_loc):
        removal_target = [0 if loc in tkwd_loc else 1 for loc in selected_loc]
        return removal_target

    def _get_content_dec_data(self, tgt, slot_dec_inp: List[int], complement_slots: List[int], complement_loc: List[int]):
        tgt_list = tgt.tolist()
        content_dec_inp = list(slot_dec_inp)
        gold_words_per_slot = defaultdict(list)
        for slot, word_loc in zip(complement_slots, complement_loc):
            gold_words_per_slot[slot].append(tgt_list[word_loc])

        num_prev_slots = 0
        target_locs = []
        target_words = []
        for slot, words in gold_words_per_slot.items():
            new_idx = slot + num_prev_slots
            content_dec_inp.insert(new_idx, self.mask_id)
            target_locs.extend([new_idx] * len(words))
            target_words.extend(words)
            num_prev_slots += 1

        for i, w in enumerate(content_dec_inp):
            if w != self.mask_id:
                target_locs.append(i)
                target_words.append(w)

        return content_dec_inp, target_locs, target_words

    def _get_slot_dec_data(self, tgt: torch.Tensor, selected_loc: List[int], complement_loc: List[int]):
        """Return input and gold targets for slot decoder."""
        slot_dec_inp = tgt.gather(dim=0, index=tgt.new_tensor(selected_loc)).tolist()

        # for each word pointed by a element of complement_loc, find the slot number of the word.
        slots = []
        for i, x in enumerate(complement_loc):
            if i == 0:
                slots.append(x)
            else:
                prev_x = complement_loc[i - 1]
                prev_slot = slots[-1]
                slots.append(x - prev_x - 1 + prev_slot)

        return slot_dec_inp, slots

    def get_weights_for_slots(self, slots: List[int]) -> List[float]:
        # Uniform weights inside every slot
        slot_counts = dict((k, len(list(g)))for k, g in itertools.groupby(slots))
        weights = [1./slot_counts[s] for s in slots]    # uniform weights for every slot
        return weights

    def get_sel_compl_locs(self, tgt: torch.LongTensor, tkwd: torch.LongTensor, selected_num: int):
        tgt_len = tgt.size()[0]
        tkwd_len = tkwd.size()[0]

        # 2. find keyword locations
        tkwd_loc = DecoupledInsTrans.get_keyword_locations_in_target(tkwd, tgt)

        # 3. and sample auxiliary words other than keywords
        remaining_loc = list(set(range(tgt_len)) - set(tkwd_loc))
        random.shuffle(remaining_loc)
        aux_loc = remaining_loc[:selected_num]
        selected_loc = sorted(aux_loc + tkwd_loc)

        # 4. return the complement words for keywords + aux. words
        complement_loc = sorted(set(range(tgt_len)) - set(selected_loc))

        return tkwd_loc, selected_loc, complement_loc

    @staticmethod
    def get_keyword_locations_in_target(tkwd, tgt):
        # 2. find keyword locations
        tgt_arr, tkwd_arr = tgt.tolist(), tkwd.tolist()
        tkwd_loc, start = [], 0
        for k in tkwd_arr:
            subarr = tgt_arr[start:]
            if k not in subarr:
                continue

            loc = start + subarr.index(k)
            tkwd_loc.append(loc)
            start = loc + 1
        tkwd_loc.sort()
        return tkwd_loc


class ConservativeL2RTransformation(TrainingTransformation):

    END_OF_SPAN_TOKEN = "<eospan>"
    MASK_TOKEN = "<mask>"

    logger = logging.getLogger(__name__)

    def example_gen(self, tkwds: List[torch.LongTensor], tgts: List[torch.LongTensor]):
        generators = [self._generate_from_example(tkwd, tgt) for tkwd, tgt in zip(tkwds, tgts)]
        yield from zip(*generators)

    def _generate_from_example(self, tkwd: torch.LongTensor, tgt: torch.LongTensor):
        tkwd_loc = DecoupledInsTrans.get_keyword_locations_in_target(tkwd, tgt)
        yield from []


if __name__ == '__main__':
    import unittest

    class UniformInsTransTest(unittest.TestCase):
        def test_weights(self):
            order = UniformInsTrans(eos_id=1)

            slots = []
            weights = order.get_weights_for_slots(slots)
            self.assertEqual(weights, [])

            slots = [random.randint(0, 100)]
            weights = order.get_weights_for_slots(slots)
            self.assertEqual(weights, [1])

            slots = [0, 1, 2, 3, 4, 5]
            weights = order.get_weights_for_slots(slots)
            self.assertEqual(weights, [1] * 6)

            slots = [0, 0, 0, 0, 0, 0]
            weights = order.get_weights_for_slots(slots)
            self.assertEqual(weights, [1/6.] * 6)

            slots = [0, 0, 0, 1, 1, 1]
            weights = order.get_weights_for_slots(slots)
            self.assertEqual(weights, [1/3.] * 6)

            slots = [0, 1, 1, 2, 3, 4]
            weights = order.get_weights_for_slots(slots)
            self.assertEqual(weights, [1, 1/2., 1/2., 1, 1, 1])

            slots = [0, 1, 1, 2, 3, 4]
            weights = order.get_weights_for_slots(slots)
            self.assertEqual(weights, [1, 1/2., 1/2., 1, 1, 1])

            slots = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
            weights = order.get_weights_for_slots(slots)
            self.assertEqual(weights, [1, 1/2., 1/2., 1/3., 1/3., 1/3., 1/4., 1/4., 1/4., 1/4.])

        def test_select(self):
            order = UniformInsTrans(eos_id=1)

            tgt = torch.tensor([3, 4, 5, 6, 7]).long()
            tkwd = torch.tensor([5, 6, 7]).long()
            selected, complement = order.get_sel_compl_locs(tgt, tkwd, 0)
            self.assertEqual(selected, [2, 3, 4])
            self.assertEqual(complement, [0, 1])

            selected, complement = order.get_sel_compl_locs(tgt, tkwd, 1)
            self.assertEqual(len(selected), 3 + 1)
            self.assertEqual(len(complement), 1)
            self.assertTrue(all(x in selected for x in [2, 3, 4]))

            selected, complement = order.get_sel_compl_locs(tgt, tkwd, 2)
            self.assertEqual(selected, list(range(5)))
            self.assertEqual(complement, [])

        def test_random_select(self):
            order = UniformInsTrans(eos_id=1)
            for i in range(10000):
                tgt_len = random.randint(5, 30)
                tgt = torch.randint(0, 1000, (tgt_len,))
                tkwd_len = random.randint(1, min(tgt_len, 5))
                all_locs = list(range(tgt_len))
                random.shuffle(all_locs)
                tkwd_locs = all_locs[:tkwd_len]
                tkwd_locs.sort()
                tkwd = tgt.gather(dim=0, index=torch.tensor(tkwd_locs))

                k = random.randint(0, tgt_len - tkwd_len) # 0 <= k <= #tgt - #tkwd, random.randint is right-inclusive
                selected, complement = order.get_sel_compl_locs(tgt, tkwd, k)

                self.assertEqual(len(selected), tkwd_len + k)
                self.assertEqual(len(selected) + len(complement), tgt_len)
                # it is possible that a keyword occurs several times in the target, which could not found in selected
                # for i in tkwd_locs:
                #     self.assertIn(i, selected)
                #     self.assertNotIn(i, complement)

                for i in selected:
                    self.assertNotIn(i, complement)

                for i in complement:
                    self.assertNotIn(i, selected)

                self.assertCountEqual(selected, set(selected))
                self.assertCountEqual(complement, set(complement))

        def test_single_example_processing(self):
            order = UniformInsTrans(eos_id=-1, use_slot_loss=True)

            for i in range(1000):
                tgt_len = 30
                tgt = torch.randint(0, 1000, (tgt_len,))
                tkwd_len = random.randint(1, 7)
                tkwd = torch.randint(0, 1000, (tkwd_len,))
                dec_inp, target_slots, target_toks, target_weights = order._process_single_example(tgt, tkwd)

                self.assertEqual(target_slots.size()[0], target_weights.size()[0])
                self.assertEqual(target_slots.size()[0], target_toks.size()[0])

                slot_number = dec_inp.size()[0] + 1
                for x in target_slots.tolist():
                    self.assertLess(x, slot_number)

                for x in dec_inp.tolist():
                    self.assertLess(x, 1000)
                    self.assertGreaterEqual(x, 0)

    unittest.main()

