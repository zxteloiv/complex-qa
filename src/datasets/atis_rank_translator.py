from typing import List, Generator, Tuple, Mapping, Optional
from trialbot.training import Registry
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from trialbot.data import Translator, NSVocabulary, START_SYMBOL, END_SYMBOL, PADDING_TOKEN

@Registry.translator('atis_rank')
class AtisRankTranslator(Translator):
    def __init__(self, max_len: int = 50):
        super().__init__()
        self.max_len = max_len
        self.namespace = ("nl", "lf")   # natural language, and logical form

    def split_nl_sent(self, sent: str):
        return sent.strip().split(" ")

    def split_lf_seq(self, seq: str):
        return seq.strip().split(" ")

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
        tgt = self.split_lf_seq(tgt)[:self.max_len]
        hyp = self.split_lf_seq(hyp)[:self.max_len]

        ns_nl, ns_lf = self.namespace
        src_toks = torch.tensor([self.vocab.get_token_index(tok, ns_nl) for tok in src])
        tgt_toks = torch.tensor([self.vocab.get_token_index(tok, ns_lf) for tok in tgt])
        hyp_toks = torch.tensor([self.vocab.get_token_index(tok, ns_lf) for tok in hyp])
        label = torch.tensor(int(is_correct))   # 1 for True, 0 for False
        instance = {"source_tokens": src_toks, "target_tokens": tgt_toks, "hyp_tokens": hyp_toks,
                    "ex_id": eid, "hyp_rank": hyp_rank, "hyp_label": label}
        return instance

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]):
        assert len(tensors) > 0
        tensor_list_by_keys = defaultdict(list)
        for instance in tensors:
            for k, tensor in instance.items():
                tensor_list_by_keys[k].append(tensor)

        # # discard batch because every source
        # return tensor_list_by_keys

        nl_padding = self.vocab.get_token_index(PADDING_TOKEN, self.namespace[0])
        lf_padding = self.vocab.get_token_index(PADDING_TOKEN, self.namespace[1])
        def pad_tensors(key: str, pad: int):
            return pad_sequence(tensor_list_by_keys[key], batch_first=True, padding_value=pad)

        batched_tensor = {"hyp_label": torch.stack(tensor_list_by_keys["hyp_label"], dim=0),
                          "source_tokens": pad_tensors("source_tokens", nl_padding),
                          "ex_id": tensor_list_by_keys["ex_id"],
                          "hyp_rank": tensor_list_by_keys["hyp_rank"],
                          "target_tokens": pad_tensors("target_tokens", lf_padding),
                          "hyp_tokens": pad_tensors("hyp_tokens", lf_padding),}

        return batched_tensor

