from typing import Generator, Tuple, List, Mapping, Optional, Any
import torch
from torch.nn.utils.rnn import pad_sequence
from trialbot.training import Registry
from trialbot.data import START_SYMBOL, END_SYMBOL, PADDING_TOKEN
from trialbot.data.translator import Translator
import re
from itertools import product
from collections import defaultdict
from functools import partial
from utils.sparql_tokenizer import split_sparql

@Registry.translator()
class CompWebQTranslator(Translator):
    def __init__(self, max_lf_len=100, max_nl_len=40):
        super().__init__()
        self.ns = ("ns_q", "ns_mq", "ns_lf")
        self.max_lf_len = max_lf_len
        self.max_nl_len = max_nl_len

    def split_question(self, q: str):
        # replace phrases with consecutive capitalized initials as anonymized entities
        q = re.sub(r"[A-Z]\w+( on| de| [A-Z]\w+)* [A-Z]\w+", "<ent>", q)
        return q.strip().split(" ")

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        mq, q, sparql = list(map(example.get, ("machine_question", "question", "sparql")))
        ns_q, ns_mq, ns_lf = self.ns
        mq_toks, q_toks = list(map(self.split_question, (mq, q)))
        sparql_toks = split_sparql(sparql)

        yield from [(ns_lf, START_SYMBOL), (ns_lf, END_SYMBOL)]
        yield from product([ns_q], q_toks[:self.max_nl_len])
        yield from product([ns_mq], mq_toks[:self.max_lf_len])
        yield from product([ns_lf], sparql_toks[:self.max_lf_len])

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        assert self.vocab is not None, "vocabulary must be set before assigning ids to instances"
        mq, q, sparql = list(map(example.get, ("machine_question", "question", "sparql")))
        ns_q, ns_mq, ns_lf = self.ns
        mq_toks = self.split_question(mq)[:self.max_nl_len]
        q_toks = self.split_question(q)[:self.max_nl_len]
        sparql_toks = [START_SYMBOL] + split_sparql(sparql)[:self.max_lf_len] + [END_SYMBOL]

        instance = {
            "mq": self._seq_word_vec(mq_toks, ns_mq),
            "q": self._seq_word_vec(q_toks, ns_q),
            "sparql": self._seq_word_vec(sparql_toks, ns_lf),
            "qid": example['ID'],
            "_raw": {
                "mq_toks": self._reconstructed_example(mq_toks, ns_mq),
                "q_toks": self._reconstructed_example(q_toks, ns_q),
                "sparql_toks": self._reconstructed_example(sparql_toks, ns_lf),
                "example": example,
            }
        }
        return instance

    def _seq_word_vec(self, seq: List[str], ns: str) -> Optional[torch.Tensor]:
        if seq is None or len(seq) == 0:
            return None
        # word tokens
        return torch.tensor([self.vocab.get_token_index(tok, ns) for tok in seq])

    def _reconstructed_example(self, seq: List[str], ns: str) -> Optional[List[str]]:
        if seq is None or len(seq) == 0:
            return None
        # word tokens
        reconstructed = []
        for tok in seq:
            tok_id = self.vocab.get_token_index(tok, ns)
            reconstructed.append(self.vocab.get_token_from_index(tok_id, ns))
        return reconstructed

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        assert len(tensors) > 0
        list_by_keys = self.list_of_dict_to_dict_of_list(tensors)

        get_ns_pad = lambda ns: self.vocab.get_token_index(PADDING_TOKEN, ns)
        pad_seq_b = lambda k, ns: pad_sequence(list_by_keys[k], batch_first=True, padding_value=get_ns_pad(ns))
        ns_q, ns_mq, ns_lf = self.ns

        batch = {
            "qid": list_by_keys["qid"],
            "mq": pad_seq_b("mq", ns_mq),
            "q": pad_seq_b("q", ns_q),
            "sparql": pad_seq_b("sparql", ns_lf),
            "_raw": list_by_keys['_raw'],
        }
        return batch

    @staticmethod
    def list_of_dict_to_dict_of_list(ld: List[Mapping[str, Any]]) -> Mapping[str, List[Any]]:
        list_by_keys = defaultdict(list)
        for d in ld:
            for k, v in d.items():
                list_by_keys[k].append(v)
        return list_by_keys



