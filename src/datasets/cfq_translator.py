from typing import List, Mapping, Generator, Tuple, Optional, Any
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from trialbot.training import Registry
from trialbot.data import Translator, START_SYMBOL, END_SYMBOL, PADDING_TOKEN
from utils.sparql_tokenizer import split_sparql
from itertools import product

@Registry.translator('cfq_seq')
class CFQSeq(Translator):
    def __init__(self):
        super().__init__()

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        key = 'sparqlPattern'
        sparql_pattern = example.get(key)
        yield from [(key, START_SYMBOL), (key, END_SYMBOL)]
        yield from product([key], split_sparql(sparql_pattern))

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        key = 'sparqlPattern'
        sparql_pattern = example.get(key)
        sparql_pattern_toks = [START_SYMBOL] + split_sparql(sparql_pattern) + [END_SYMBOL]
        instance = {
            "sparqlPattern": self._seq_word_vec(sparql_pattern_toks, key),
            "_raw": {
                "reconstructed_sparql_pattern": self._reconstructed_example(sparql_pattern_toks, key),
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
        list_by_keys = list_of_dict_to_dict_of_list(tensors)
        get_ns_pad = lambda ns: self.vocab.get_token_index(PADDING_TOKEN, ns)
        pad_seq_b = lambda k, ns: pad_sequence(list_by_keys[k], batch_first=True, padding_value=get_ns_pad(ns))

        batch = {
            "sparqlPattern": pad_seq_b("sparqlPattern", "sparqlPattern"),
            "_raw": list_by_keys['_raw'],
        }
        return batch

def list_of_dict_to_dict_of_list(ld: List[Mapping[str, Any]]) -> Mapping[str, List[Any]]:
    list_by_keys = defaultdict(list)
    for d in ld:
        for k, v in d.items():
            list_by_keys[k].append(v)
    return list_by_keys

@Registry.translator('lark')
class LarkTranslator(Translator):
    def __init__(self):
        super().__init__()
        ns_nt, ns_t, ns_et = 'nonterminal', 'terminal_category', 'exact_terminal'
        self.ns_nt, self.ns_t, self.ns_et = ns_nt, ns_t, ns_et

    def _read_s(self, symbol):
        return symbol['fidelity'], symbol['token'], symbol['exact_token']

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        ns_nt, ns_t, ns_et = self.ns_nt, self.ns_t, self.ns_et
        lhs, rhs_seq = example
        yield ns_nt, lhs
        yield from product([ns_t], [START_SYMBOL, END_SYMBOL])
        for symbol in rhs_seq:
            exactitude, token, exact_token = self._read_s(symbol)
            if exactitude > 0:
                yield ns_t, token
                if exactitude == 2:
                    yield ns_et, exact_token
            else:
                yield ns_nt, token

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        ns_nt, ns_t, ns_et = self.ns_nt, self.ns_t, self.ns_et
        lhs, rhs_seq = example
        get_tokid = self.vocab.get_token_index
        # start and end symbols belong to terminal category namespace
        rhs_fi = [1] + [s['fidelity'] for s in rhs_seq] + [1]
        d_seq = [get_tokid(lhs, ns_nt), get_tokid(START_SYMBOL, ns_t)]
        d_seq += [get_tokid(token, ns_nt if tofi == 0 else ns_t) for tofi, token, _ in map(self._read_s, rhs_seq)]
        d_seq += [get_tokid(END_SYMBOL, ns_t)]

        tensor_derivation_seq = torch.tensor(d_seq)
        tensor_fidelity = torch.tensor(rhs_fi)
        return {'derivation_tree': tensor_derivation_seq, 'token_fidelity': tensor_fidelity}

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        assert len(tensors) > 0
        list_by_keys = list_of_dict_to_dict_of_list(tensors)
        pad_id = self.vocab.get_token_index(PADDING_TOKEN, self.ns_t)
        pad_seq_b = lambda k: pad_sequence(list_by_keys[k], batch_first=True, padding_value=pad_id)
        batch = {
            "derivation_tree": pad_seq_b("derivation_tree").unsqueeze(1),
            "token_fidelity": pad_seq_b("token_fidelity").unsqueeze(1),
        }
        return batch
