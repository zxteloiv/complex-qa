from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Iterable
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from .field import FieldAwareTranslator, Field
from .seq_field import SeqField
import lark

class GrammarPatternSeqField(SeqField):
    @classmethod
    def _get_rule_str(cls, subtree):
        rule = [subtree.data]  # LHS
        for c in subtree.children:
            rule.append(cls._get_token_str(c))
        rule_str = ' '.join(rule)
        return rule_str

    @classmethod
    def _get_token_str(cls, tok):
        Tree, Token = lark.Tree, lark.Token
        if isinstance(tok, Tree):
            return tok.data

        elif isinstance(tok, Token):
            return tok.type

        else:
            return tok

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        Tree, Token = lark.Tree, lark.Token
        tree: Tree = example.get(self.source_key)
        assert tree is not None
        for subtree in tree.iter_subtrees_topdown():
            rule_str = self._get_rule_str(subtree)
            yield self.ns, rule_str

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        Tree, Token = lark.Tree, lark.Token
        tree: Tree = example.get(self.source_key)
        assert tree is not None
        rule_id = [self.vocab.get_token_index(self._get_rule_str(st), namespace=self.ns) for st in tree.iter_subtrees_topdown()]
        rule_seq_tensor = torch.tensor(rule_id)
        return {self.renamed_key: rule_seq_tensor}

class GrammarModEntSeqField(GrammarPatternSeqField):
    @classmethod
    def _get_token_str(cls, tok):
        Tree, Token = lark.Tree, lark.Token
        if isinstance(tok, Tree):
            return tok.data

        elif isinstance(tok, Token):
            if tok.type == "PNAME_LN":
                return tok.value

            else:
                return tok.type
        else:
            return tok


class MidOrderTraversalField(Field):
    def __init__(self, tree_key: str,
                 namespaces: Iterable[str],
                 output_keys: Optional[List[str]] = None,
                 padding: int = 0,
                 max_seq_len: int = 0):
        super().__init__()
        self.tree_key = tree_key
        self.namespaces = list(namespaces) or ('nonterminal', 'terminal_category', 'terminal')
        self.output_keys = output_keys or ('rule', 'is_nt', 'is_ending')
        self.padding = padding
        self.max_seq_len = max_seq_len

    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        output = dict()
        for k in self.output_keys:
            tensor_list = tensors_by_keys[k]
            batched = pad_sequence(tensor_list, batch_first=True, padding_value=self.padding)
            output[k] = batched
        return output

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        tree: lark.Tree = example.get(self.tree_key)
        ns_nt, ns_t, ns_et = self.namespaces
        for subtree in tree.iter_subtrees_topdown():
            yield ns_nt, subtree.data
            for c in subtree.children:
                if isinstance(c, lark.Token):
                    yield ns_t, c.type
                    yield ns_et, c.value

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        Tree, Token = lark.Tree, lark.Token
        tree: Tree = example.get(self.tree_key)
        assert tree is not None

        tokid = self.vocab.get_token_index
        key_rule, key_depth_stop, key_horizontal_stop = self.output_keys
        ns_nt, ns_t, ns_et = self.namespaces

        left_most_derivation = defaultdict(list)
        for subtree in tree.iter_subtrees_topdown():
            lhs = tokid(subtree.data, ns_nt)
            # for any token, the rhs contains its category name
            rhs = [tokid(s.data, ns_nt) if isinstance(s, Tree) else tokid(s.type, ns_t) for s in subtree.children]

            rule = [lhs] + rhs
            is_nt = [1] + [1 if isinstance(s, Tree) else 0 for s in subtree.children]
            is_ending = [0] * len(is_nt)
            is_ending[-1] = 1

            left_most_derivation[key_rule].extend(rule)
            left_most_derivation[key_depth_stop].extend(is_nt)
            left_most_derivation[key_horizontal_stop].extend(is_ending)

        output = dict()
        for k, l in left_most_derivation.items():
            output[k] = torch.tensor(l)

        return output

