from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
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

def list_of_dict_to_dict_of_list(ld: List[Mapping[str, Any]]) -> Mapping[str, List[Any]]:
    list_by_keys = defaultdict(list)
    for d in ld:
        for k, v in d.items():
            list_by_keys[k].append(v)
    return list_by_keys

# namespaces definition and the corresponding fidelity
PARSE_TREE_NS = (NS_NT, NS_T, NS_ET) = ('nonterminal', 'terminal_category', 'terminal')
NS_FI = (NS_NT_FI, NS_T_FI, NS_ET_FI) = (0, 1, 2)

@Registry.translator('lark')
class LarkTranslator(Translator):
    def _read_s(self, symbol):
        return symbol['fidelity'], symbol['token'], symbol['exact_token']

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        lhs, rhs_seq = example
        yield from product([NS_NT], [lhs, START_SYMBOL, END_SYMBOL])
        for symbol in rhs_seq:
            exactitude, token, _ = self._read_s(symbol)
            if exactitude > 0:
                yield NS_T, token
            else:
                yield NS_NT, token

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        lhs, rhs_seq = example
        tokid = self.vocab.get_token_index
        # start and end symbols are nonterminals
        rhs_fi = [NS_NT_FI] + [s['fidelity'] for s in rhs_seq] + [NS_NT_FI]
        symbols = [tokid(s, NS_NT) for s in (lhs, START_SYMBOL)]
        symbols += [tokid(token, NS_NT if tofi == 0 else NS_T) for tofi, token, _ in map(self._read_s, rhs_seq)]
        symbols += [tokid(END_SYMBOL, NS_NT)]

        tensor_derivation_seq = torch.tensor(symbols)
        tensor_fidelity = torch.tensor(rhs_fi)
        return {'derivation_tree': tensor_derivation_seq, 'token_fidelity': tensor_fidelity}

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        assert len(tensors) > 0
        list_by_keys = list_of_dict_to_dict_of_list(tensors)

        # both seq is actually padded by 0, simultaneously indicating the PAD_TOK of the NS_NT namespace.
        token_pad_id = self.vocab.get_token_index(PADDING_TOKEN, NS_NT)
        pad_seq_b = lambda k, pad_id: pad_sequence(list_by_keys[k], batch_first=True, padding_value=pad_id)
        batch = {
            "derivation_tree": pad_seq_b("derivation_tree", token_pad_id).unsqueeze(1),
            "token_fidelity": pad_seq_b("token_fidelity", NS_NT_FI).unsqueeze(1),
        }
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

@Registry.translator('cfq_mod_flat_tree_qa')
class CFQFlatDerivations(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False,),
            MidOrderTraversalField(tree_key='sparqlPatternModEntities_tree',
                                   namespaces=PARSE_TREE_NS)
        ])

class _CFQTreeLM(FieldAwareTranslator):
    def __init__(self, tree_key: str):
        super().__init__(field_list=[
            SeqField(source_key='questionPatternModEntities', renamed_key='source_tokens', add_start_end_toks=False, )
        ])
        self.tree_key = tree_key

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        yield from super().generate_namespace_tokens(example)
        tree: lark.Tree = example.get(self.tree_key)
        for subtree in tree.iter_subtrees_topdown():
            yield NS_NT, subtree.data
            yield from product([NS_NT], [START_SYMBOL, END_SYMBOL])
            for c in subtree.children:
                if isinstance(c, lark.Token):
                    yield NS_T, c.type
                    yield NS_ET, c.value

    def to_tensor(self, example) -> Mapping[str, Any]:
        output = dict(super().to_tensor(example))
        Tree, Token = lark.Tree, lark.Token
        tree: Tree = example.get(self.tree_key)
        derivation_tree: List[List[int]] = []
        token_fidelity: List[List[int]] = []
        tokid = self.vocab.get_token_index
        for subtree in tree.iter_subtrees_topdown():
            lhs = tokid(subtree.data, NS_NT)
            rhs = [tokid(s.data, NS_NT) if isinstance(s, Tree) else tokid(s.type, NS_T) for s in subtree.children]
            fidelity = [NS_NT_FI if isinstance(s, Tree) else NS_T_FI for s in subtree.children]
            # the fidelity vector MUST NOT contain the fidelity of LHS, which is always NS_NT_FI
            rule = [lhs] + [tokid(START_SYMBOL, NS_NT)] + rhs + [tokid(END_SYMBOL, NS_NT)]
            rule_fi = [NS_NT_FI] + fidelity + [NS_NT_FI]
            derivation_tree.append(rule)
            token_fidelity.append(rule_fi)

        output.update(derivation_tree=derivation_tree, token_fidelity=token_fidelity)
        return output

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        output = dict(super().batch_tensor(tensors))

        list_by_keys = list_of_dict_to_dict_of_list(tensors)
        tree_list: List[List[List[int]]] = list_by_keys['derivation_tree']
        tofi_list: List[List[List[int]]] = list_by_keys['token_fidelity']
        max_derivation_num = max(len(instance) for instance in tree_list)
        max_symbol_num = max(len(rule) for instance in tree_list for rule in instance)

        pad_id = self.vocab.get_token_index(PADDING_TOKEN, NS_NT)
        empty_rule = [pad_id] * max_symbol_num
        empty_fidelity = [NS_NT_FI] * (max_symbol_num - 1)

        tree_batch = []
        fidelity_batch = []
        for tree, tree_fi in zip(tree_list, tofi_list):
            if len(tree) < max_derivation_num:
                tree.extend(empty_rule for _ in range(max_derivation_num - len(tree)))
                tree_fi.extend(empty_fidelity for _ in range(max_derivation_num - len(tree_fi)))

            derivations = []
            derivations_fidelity = []
            for rule, rule_fi in zip(tree, tree_fi):
                if len(rule) < max_symbol_num:
                    rule.extend(pad_id for _ in range(len(empty_rule) - len(rule)))
                    rule_fi.extend(NS_NT_FI for _ in range(len(empty_fidelity) - len(rule_fi)))
                derivations.append(torch.tensor(rule))
                derivations_fidelity.append(torch.tensor(rule_fi))

            # derivations: (max_derivation, max_symbol), after stacking
            # derivations_fidelity: (max_derivation, max_symbol - 1), after stacking
            tree_batch.append(torch.stack(derivations))
            fidelity_batch.append(torch.stack(derivations_fidelity))

        # tree_batch: (batch, max_derivation, max_symbol)
        # fidelity_batch: (batch, max_derivation, max_symbol - 1)
        tree_batch = torch.stack(tree_batch)
        fidelity_batch = torch.stack(fidelity_batch)

        output.update({ "derivation_tree": tree_batch, "token_fidelity": fidelity_batch, })
        return output

# @Registry.translator('cfq_pattern_tree')
# class CFQPatternTree(_CFQTreeLM):
#     def __init__(self):
#         super().__init__('sparqlPattern_tree')

@Registry.translator('cfq_mod_tree')
class CFQModTree(_CFQTreeLM):
    def __init__(self):
        super().__init__('sparqlPatternModEntities_tree')

# @Registry.translator('cfq_complete_tree')
# class CFQCompleteTree(_CFQTreeLM):
#     def __init__(self):
#         super().__init__('sparql_tree')
