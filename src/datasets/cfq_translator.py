from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Dict
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from trialbot.training import Registry
from trialbot.data import Translator, START_SYMBOL, END_SYMBOL, PADDING_TOKEN
from utils.sparql_tokenizer import split_sparql
from itertools import product
import lark
from .field import FieldAwareTranslator
from .seq_field import SeqField
from .cfq_fields import MidOrderTraversalField, GrammarPatternSeqField, GrammarModEntSeqField

@Registry.translator('cfq_seq_mod_ent_qa')
class CFQSeq(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='sparqlPatternModEntities', renamed_key="target_tokens", split_fn=split_sparql,),
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,)
        ])

# namespaces definition and the corresponding fidelity
PARSE_TREE_NS = (NS_NT, NS_T, NS_ET) = ('nonterminal', 'terminal_category', 'terminal')
NS_FI = (NS_NT_FI, NS_T_FI, NS_ET_FI) = (0, 1, 2)
UNIFIED_TREE_NS = ('symbol', 'exact_token')

@Registry.translator('u_lark')
class UnifiedLarkTranslator(Translator):
    def _read_s(self, symbol):
        return symbol['fidelity'], symbol['token'], symbol['exact_token']

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        lhs, rhs_seq = example
        ns = UNIFIED_TREE_NS[0]
        yield ns, lhs
        yield from product(UNIFIED_TREE_NS, [START_SYMBOL])
        yield from ((ns, symbol) for _, symbol, _ in map(self._read_s, rhs_seq))

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        lhs, rhs_seq = example
        tokid = lambda t: self.vocab.get_token_index(t, UNIFIED_TREE_NS[0])

        # insert a default start token at the beginning in case of the empty expansion
        symbols, has_children = [tokid(START_SYMBOL)], [0]
        for tofi, token, _ in map(self._read_s, rhs_seq):
            symbols.append(tokid(token))
            has_children.append(1 if tofi == NS_NT_FI else 0)

        is_a_valid_successor = [1] * len(symbols)

        # each rule rhs is a tensor of (3, rhs_seq)
        rhs_tensor = torch.tensor(list(zip(symbols, has_children, is_a_valid_successor))).t()
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

@Registry.translator('cfq_tranx_pattern_qa')
class TranXStyleTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,),
            GrammarPatternSeqField(source_key='sparqlPatternModEntities_tree', renamed_key="target_tokens",
                                   namespace="modent_rule_seq", split_fn=split_sparql, ),
        ])

@Registry.translator('cfq_tranx_mod_ent_qa')
class TranXModEntTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,),
            GrammarModEntSeqField(source_key='sparqlPatternModEntities_tree', renamed_key="target_tokens",
                                  namespace="modent_rule_seq", split_fn=split_sparql, ),
        ])

@Registry.translator('cfq_flat_pda_mod_ent_qa')
class CFQFlatDerivations(FieldAwareTranslator):
    def __init__(self, max_derivation_symbols: int = 11):
        super().__init__(field_list=[
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,),
            MidOrderTraversalField(tree_key='sparqlPatternModEntities_tree',
                                   namespaces=UNIFIED_TREE_NS,
                                   max_derivation_symbols=max_derivation_symbols,
                                   )
        ])

