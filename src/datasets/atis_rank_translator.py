from typing import List, Generator, Tuple, Mapping, Optional
from trialbot.training import Registry
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from trialbot.data import Translator, NSVocabulary, START_SYMBOL, END_SYMBOL, PADDING_TOKEN
import logging
from itertools import product

@Registry.translator('atis_rank')
class AtisRankTranslator(Translator):
    def __init__(self, max_len: int = 60):
        super().__init__()
        self.max_len = max_len
        self.namespace = ("nl", "lf")   # natural language, and logical form
        self.logger = logging.getLogger('translator')
        self.add_special_token = False

    def turn_special_token(self, on=True):
        self.add_special_token = on

    def split_nl_sent(self, sent: str):
        return sent.strip().split(" ")

    def split_lf_seq(self, seq: str):
        return seq.strip().split(" ")

    def _add_tokens(self, seq: List):
        return [START_SYMBOL] + seq + [END_SYMBOL]

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        src, tgt = list(map(example.get, ("src", "tgt")))
        # src is in list, but but tgt is still a sentence
        tgt = self.split_lf_seq(tgt)[:self.max_len]

        ns_nl, ns_lf = self.namespace
        default_toks = [(ns_nl, START_SYMBOL), (ns_nl, END_SYMBOL), (ns_lf, START_SYMBOL), (ns_lf, END_SYMBOL)]
        yield from default_toks

        for tok in src:
            yield ns_nl, tok

        for tok in tgt:
            yield ns_lf, tok

    def to_tensor(self, example):
        assert self.vocab is not None, "vocabulary must be set before assigning ids to instances"

        eid, hyp_rank, src, tgt, hyp, is_correct = list(
            map(example.get, ("ex_id", "hyp_rank", "src", "tgt", "hyp", "is_correct"))
        )
        try:
            tgt = self.split_lf_seq(tgt)[:self.max_len]
            hyp = self.split_lf_seq(hyp)[:self.max_len]

            if self.add_special_token:
                src = self._add_tokens(src)
                tgt = self._add_tokens(tgt)
                hyp = self._add_tokens(hyp)

        except:
            self.logger.warning("Ignored invalid hypothesis %d-%d." % (eid, hyp_rank))
            tgt = hyp = None

        ns_nl, ns_lf = self.namespace
        src_toks = torch.tensor([self.vocab.get_token_index(tok, ns_nl) for tok in src])
        tgt_toks = torch.tensor([self.vocab.get_token_index(tok, ns_lf) for tok in tgt]) if tgt else None
        hyp_toks = torch.tensor([self.vocab.get_token_index(tok, ns_lf) for tok in hyp]) if hyp else None
        label = torch.tensor(int(is_correct))   # 1 for True, 0 for False
        instance = {"source_tokens": src_toks, "target_tokens": tgt_toks, "hyp_tokens": hyp_toks,
                    "ex_id": eid, "hyp_rank": hyp_rank, "hyp_label": label, "_raw": example}
        return instance

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]):
        assert len(tensors) > 0
        list_by_keys = defaultdict(list)
        for instance in tensors:
            if instance['hyp_tokens'] is None:
                continue    # discard invalid data
            for k, tensor in instance.items():
                list_by_keys[k].append(tensor)

        # # discard batch because every source
        # return list_by_keys

        nl_padding = self.vocab.get_token_index(PADDING_TOKEN, self.namespace[0])
        lf_padding = self.vocab.get_token_index(PADDING_TOKEN, self.namespace[1])
        def pad_tensors(key: str, pad: int):
            return pad_sequence(list_by_keys[key], batch_first=True, padding_value=pad)

        batched_tensor = {"hyp_label": torch.stack(list_by_keys["hyp_label"], dim=0),
                          "source_tokens": pad_tensors("source_tokens", nl_padding),
                          "ex_id": list_by_keys["ex_id"],
                          "hyp_rank": torch.stack(list_by_keys["hyp_rank"], dim=0),
                          "target_tokens": pad_tensors("target_tokens", lf_padding),
                          "hyp_tokens": pad_tensors("hyp_tokens", lf_padding),
                          "_raw": list_by_keys["_raw"],
                          }

        return batched_tensor

@Registry.translator('atis_rank_char')
class AtisRankChTranslator(AtisRankTranslator):
    def __init__(self, max_len: int = 60):
        super().__init__(max_len)
        self.namespace = ("nl", "lf", "nlch", "lfch")

    def _add_tokens(self, seq: List):
        return [START_SYMBOL] + seq + [END_SYMBOL]

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        src, tgt = list(map(example.get, ("src", "tgt")))
        # src is in list, but but tgt is still a sentence
        tgt = self.split_lf_seq(tgt)[:self.max_len]

        ns_nl, ns_lf, ns_nlch, ns_lfch = self.namespace
        yield from product(self.namespace, [START_SYMBOL, END_SYMBOL])

        for tok in src:
            yield ns_nl, tok
            yield from product([ns_nlch], tok)  # iterate over the token for each character

        for tok in tgt:
            yield ns_lf, tok
            yield from product([ns_lfch], tok)  # iterate over the token for each character

    def to_tensor(self, example):
        assert self.vocab is not None, "vocabulary must be set before assigning ids to instances"

        eid, hyp_rank, src, tgt, hyp, is_correct = list(
            map(example.get, ("ex_id", "hyp_rank", "src", "tgt", "hyp", "is_correct"))
        )
        try:
            tgt = self.split_lf_seq(tgt)[:self.max_len]
            hyp = self.split_lf_seq(hyp)[:self.max_len]

        except:
            self.logger.warning("Ignored invalid hypothesis %d-%d." % (eid, hyp_rank))
            tgt = hyp = None

        ns_nl, ns_lf, ns_nlch, ns_lfch = self.namespace

        if self.add_special_token:
            src = self._add_tokens(src)
            tgt = self._add_tokens(tgt) if tgt else None
            hyp = self._add_tokens(hyp) if hyp else None

        # for every seq, get the list of characters of each word
        src_chs_list = self._seq_char_vecs(src, ns_nlch)
        tgt_chs_list = self._seq_char_vecs(tgt, ns_lfch)
        hyp_chs_list = self._seq_char_vecs(hyp, ns_lfch)

        src_toks = self._seq_word_vec(src, ns_nl)
        tgt_toks = self._seq_word_vec(tgt, ns_lf)
        hyp_toks = self._seq_word_vec(hyp, ns_lf)

        label = torch.tensor(int(is_correct))   # 1 for True, 0 for False

        instance = {"source_tokens": src_toks,
                    "target_tokens": tgt_toks,
                    "hyp_tokens": hyp_toks,
                    "src_char_list": src_chs_list,
                    "tgt_char_list": tgt_chs_list,
                    "hyp_char_list": hyp_chs_list,
                    "ex_id": eid,
                    "hyp_rank": hyp_rank,
                    "hyp_label": label,
                    "_raw": example}
        return instance

    def _seq_char_vecs(self, seq: List[str], ns: str) -> Optional[List[torch.LongTensor]]:
        """get the list of char vector of each word"""
        if seq is None or len(seq) == 0:
            return None

        # list of char tokens
        all_token_chs: List[torch.LongTensor] = []
        for tok in seq:
            if tok in (START_SYMBOL, END_SYMBOL):
                chs = self._add_tokens([PADDING_TOKEN])
            else:
                chs = self._add_tokens(list(tok))

            chs_ids = torch.tensor(list(self.vocab.get_token_index(ch, ns) for ch in chs))
            all_token_chs.append(chs_ids)

        return all_token_chs

    def _seq_word_vec(self, seq: List[str], ns: str) -> Optional[torch.Tensor]:
        if seq is None or len(seq) == 0:
            return None
        # word tokens
        return torch.tensor([self.vocab.get_token_index(tok, ns) for tok in seq])

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]):
        assert len(tensors) > 0
        list_by_keys = defaultdict(list)
        for instance in tensors:
            if any(x is None or len(x) == 0    # some hypothesis has no tokens
                   for x in map(instance.get,
                                ['hyp_tokens', 'hyp_char_list',
                                 'source_tokens', 'src_char_list',
                                 'target_tokens', 'tgt_char_list',
                                 ])):
                continue    # discard invalid data
            for k, tensor in instance.items():
                list_by_keys[k].append(tensor)

        # # discard batch because every source
        # return list_by_keys

        ns_nl, ns_lf, ns_nlch, ns_lfch = self.namespace
        nl_pad = self.vocab.get_token_index(PADDING_TOKEN, ns_nl)
        lf_pad = self.vocab.get_token_index(PADDING_TOKEN, ns_lf)
        nlch_pad = self.vocab.get_token_index(PADDING_TOKEN, ns_nlch)
        lfch_pad = self.vocab.get_token_index(PADDING_TOKEN, ns_lfch)

        def pad_words(key: str, pad: int):
            return pad_sequence(list_by_keys[key], batch_first=True, padding_value=pad)

        def pad_chars(key: str, pad: int) -> torch.Tensor:
            ptpad = torch.nn.functional.pad
            batch: List[List[torch.Tensor]] = list_by_keys[key]
            max_word_count = max(len(e) for e in batch)
            max_char_size = max(word.size()[0] for example in batch for word in example)

            b_tensors = []
            for i, example in enumerate(batch):
                e_tensors = []
                for j, word in enumerate(example):
                    # padded_word: (C,)
                    padded_word = ptpad(word, [0, max_char_size - word.size()[0]], mode="constant", value=pad)
                    e_tensors.append(padded_word)

                if len(e_tensors) == 0:
                    e_tensors.append(torch.full((max_word_count, max_char_size), pad))

                e_tensor = torch.stack(e_tensors, dim=0)  # (words, C)
                padded_e = ptpad(e_tensor, [0, 0, 0, max_word_count - len(example)])    # (W, C)
                b_tensors.append(padded_e)
            b_tensor = torch.stack(b_tensors, dim=0) # (batch, W, C)
            return b_tensor

        batched_tensor = {"hyp_label": torch.stack(list_by_keys["hyp_label"], dim=0),
                          "ex_id": list_by_keys["ex_id"],
                          "hyp_rank": torch.stack(list_by_keys["hyp_rank"], dim=0),

                          "source_tokens": pad_words("source_tokens", nl_pad),
                          "target_tokens": pad_words("target_tokens", lf_pad),
                          "hyp_tokens": pad_words("hyp_tokens", lf_pad),

                          "src_char_ids": pad_chars("src_char_list", nlch_pad),
                          "tgt_char_ids": pad_chars("tgt_char_list", lfch_pad),
                          "hyp_char_ids": pad_chars("hyp_char_list", lfch_pad),

                          "_raw": list_by_keys["_raw"],
                          }

        assert batched_tensor['hyp_tokens'].size() == batched_tensor['hyp_char_ids'].size()[:-1]
        assert batched_tensor['target_tokens'].size() == batched_tensor['tgt_char_ids'].size()[:-1]
        assert batched_tensor['source_tokens'].size() == batched_tensor['src_char_ids'].size()[:-1]

        return batched_tensor
