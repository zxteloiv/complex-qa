from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Dict
from collections import defaultdict
from trialbot.data import START_SYMBOL, END_SYMBOL, PADDING_TOKEN
from trialbot.data.translator import Translator
import torch
from .field import FieldAwareTranslator
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from itertools import product
import lark
import logging

class UnifiedLarkTranslator(Translator):
    def __init__(self, ns):
        super().__init__()
        self.ns = ns

    def _read_s(self, symbol):
        return symbol['fidelity'], symbol['token'], symbol['exact_token']

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        lhs, rhs_seq = example
        ns = self.ns[0]
        yield ns, lhs
        yield from product(self.ns, [START_SYMBOL])
        yield from ((ns, symbol) for _, symbol, _ in map(self._read_s, rhs_seq))

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        lhs, rhs_seq = example
        tokid = lambda t: self.vocab.get_token_index(t, self.ns[0])

        # insert a default start token at the beginning in case of the empty expansion
        symbols, has_children = [tokid(START_SYMBOL)], [0]
        for tofi, token, _ in map(self._read_s, rhs_seq):
            symbols.append(tokid(token))
            has_children.append(1 if tofi == 0 else 0)

        has_successor = [1] * len(symbols)
        has_successor[-1] = 0
        mask = [1] * len(symbols)

        # each rule rhs is a tensor of (4, rhs_seq)
        rhs_tensor = torch.tensor(list(zip(symbols, has_children, has_successor, mask))).t()
        return {'lhs_token': tokid(lhs), 'rhs_tensor': rhs_tensor}

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        assert len(tensors) > 0
        rule_lookup_table = defaultdict(list)
        max_rule_count = max_rhs_length = 0
        for rule in tensors:
            lhs_id, rhs_tensor = rule['lhs_token'], rule['rhs_tensor']
            rule_lookup_table[lhs_id].append(rhs_tensor)
            max_rule_count = max(max_rule_count, len(rule_lookup_table[lhs_id]))
            max_rhs_length = max(max_rhs_length, rhs_tensor.size()[-1])

        # every rule (3, varying_rhs_len) will be padded into (3, max_rhs_len)
        # every lhs will be padded into (max_rule_count, 3, max_rhs_len)
        batch = dict()
        for lhs, candidates in rule_lookup_table.items():
            padded_candidates = [pad(r, [0, max_rhs_length - r.size()[-1]]) for r in candidates]
            while len(padded_candidates) < max_rule_count:
                padded_candidates.append(torch.zeros_like(padded_candidates[-1]))
            batch[lhs] = torch.stack(padded_candidates, dim=0)

        del rule_lookup_table
        return batch

class LarkExactTokenReader:
    def __init__(self, ns, vocab, ignored_toks=None):
        super().__init__()
        self.ns = ns
        self.vocab = vocab
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ignored_toks = ignored_toks if ignored_toks is not None else (0, 1)

    @staticmethod
    def _read_s(symbol):
        return symbol['fidelity'], symbol['token'], symbol['exact_token']

    def generate_symbol_token_pair(self, example):
        lhs, rhs_seq = example

        symbolid = lambda t: self.vocab.get_token_index(t, self.ns[0])
        extoknid = lambda t: self.vocab.get_token_index(t, self.ns[1])

        yield symbolid(START_SYMBOL), extoknid(START_SYMBOL)

        for tofi, symbol, exact_token in map(self._read_s, rhs_seq):
            if tofi == 0:
                continue

            sid = symbolid(symbol)
            tid = extoknid(exact_token.lower() if isinstance(exact_token, str) else exact_token)

            if sid in self.ignored_toks or tid in self.ignored_toks:
                logging.warning(f'maybe OOV found: "{symbol}" is {sid}, "{exact_token}" is {tid}')

            else:
                yield sid, tid

    def merge_the_pairs(self, pair_set):
        valid_opts = defaultdict(list)
        for (k, v) in pair_set:
            valid_opts[k].append(v)
        return valid_opts
