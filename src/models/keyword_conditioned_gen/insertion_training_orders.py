from typing import Callable, List, Generator, Iterable, Union, Tuple
import torch
import torch.nn
import itertools
import random

_DECODER_INPUT = List[torch.LongTensor]
_NUMBER_ARRAY = torch.LongTensor
_PRED_LOC = _NUMBER_ARRAY # a batch of indices to slots
_PRED_TOK = _NUMBER_ARRAY # a batch of indices to vocabulary set
_PRED_WEIGHT = torch.Tensor # a batch of indices to vocabulary set

INSERTION_ORDER = Generator[Tuple[_DECODER_INPUT, _PRED_LOC, _PRED_TOK, _PRED_WEIGHT], None, None]

class TrainingOrder:
    """
    A general interface that specifies how an insertion transformer should be trained.
    """
    def __call__(self, *args, **kwargs):
        yield from self.example_order_gen(*args, **kwargs)

    def example_order_gen(self, *args, **kwargs) -> INSERTION_ORDER:
        raise NotImplemented


class L2ROrder(TrainingOrder):
    def __init__(self, eos_id: int):
        self.eos_id: int = eos_id

    def example_order_gen(self,
                          target_tokens: List[torch.LongTensor]
                          ) -> INSERTION_ORDER:
        """
        Yields decoder training samples from left to right.

        :param target_tokens: [(tgt_len,)]
        :return: Generator tuple of (decoder input, gold tokens, gold locations)
        """
        assert all(x.ndimension() == 1 for x in target_tokens)
        max_len = max(x.size()[0] for x in target_tokens)
        batch_size = len(target_tokens)

        sample_tensor = target_tokens[0]

        for l in range(max_len):
            dec_inp = []
            toks = []
            locs = [l] * batch_size
            weights = [1] * batch_size

            for x in target_tokens:
                dec_inp.append(x[:l])
                toks.append(self.eos_id if l >= x.size()[0] else x[l])

            # dec_inp: list of torch.LongTensor
            # toks: list of int
            # locs: list of int
            yield dec_inp, toks, locs, weights

class KwdUniformOrder(TrainingOrder):
    def __init__(self, eos_id, use_slot_loss=True):
        self.eos_id = eos_id
        self._use_slot_loss = use_slot_loss

    END_OF_SPAN_TOKEN = "@End-of-Span@"

    def example_order_gen(self, tkwds: List[torch.LongTensor], tgts: List[torch.LongTensor]):
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


class KwdBSTOrder(KwdUniformOrder):

    def __init__(self, eos_id, use_slot_loss=False, tao=1., reverse=False):
        super(KwdBSTOrder, self).__init__(eos_id, use_slot_loss)
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


if __name__ == '__main__':
    import unittest

    class KwdUniformOrderTest(unittest.TestCase):
        def test_weights(self):
            order = KwdUniformOrder(eos_id=1)

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
            order = KwdUniformOrder(eos_id=1)

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
            order = KwdUniformOrder(eos_id=1)
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
            order = KwdUniformOrder(eos_id=-1, use_slot_loss=True)

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

