from typing import List, Generator, Tuple, Mapping, Optional
from trialbot.training import Registry
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from trialbot.data import Translator, NSVocabulary, START_SYMBOL, END_SYMBOL, PADDING_TOKEN
import logging

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
                          "hyp_rank": list_by_keys["hyp_rank"],
                          "target_tokens": pad_tensors("target_tokens", lf_padding),
                          "hyp_tokens": pad_tensors("hyp_tokens", lf_padding),
                          "_raw": list_by_keys["_raw"],
                          }

        return batched_tensor

