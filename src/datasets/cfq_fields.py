from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Iterable, Union
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from .field import FieldAwareTranslator, Field
from .seq_field import SeqField
import lark
from trialbot.data import START_SYMBOL

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
                 max_derivation_symbols: int,
                 output_keys: Optional[List[str]] = None,
                 padding: int = 0,
                 grammar_token_generation: bool = False,
                 ):
        super().__init__()
        self.tree_key = tree_key
        self.namespaces = list(namespaces) or ('symbols', 'exact_token')
        self.output_keys = output_keys or (
            'rhs_symbols', 'parental_growth', 'rhs_exact_tokens', 'mask', 'target_tokens'
        )
        self.padding = padding
        self.max_derivation_symbols = max_derivation_symbols
        self.grammar_token_generation = grammar_token_generation

    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        output = dict()
        for k in self.output_keys:
            tensor_list = tensors_by_keys[k]
            batched = pad_sequence(tensor_list, batch_first=True, padding_value=self.padding)
            output[k] = batched
        return output

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        tree: lark.Tree = example.get(self.tree_key)
        ns_s, ns_et = self.namespaces
        for subtree in tree.iter_subtrees_topdown():
            if self.grammar_token_generation:
                yield ns_s, subtree.data
                yield ns_s, START_SYMBOL
            yield ns_et, START_SYMBOL
            for c in subtree.children:
                if isinstance(c, lark.Token):
                    if self.grammar_token_generation:
                        yield ns_s, c.type
                    yield ns_et, c.value.lower()

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        Tree, Token = lark.Tree, lark.Token
        tree: Tree = example.get(self.tree_key)
        assert tree is not None

        tokid = self.vocab.get_token_index
        ns_s, ns_et = self.namespaces

        left_most_derivation = defaultdict(list)
        for subtree in tree.iter_subtrees_topdown():
            children: List[Union[Tree, Token]] = subtree.children

            rhs_symbol, rhs_exact_token, parental_growth = [tokid(START_SYMBOL, ns_s)], [tokid(START_SYMBOL, ns_et)], [0]
            for s in children:
                rhs_symbol.append(tokid(s.data, ns_s) if isinstance(s, Tree) else tokid(s.type, ns_s))
                rhs_exact_token.append(tokid(s.value.lower(), ns_et) if isinstance(s, Token) else self.padding)
                parental_growth.append(1 if isinstance(s, Tree) else 0)

            mask = [1] * len(rhs_symbol)

            if len(rhs_symbol) < self.max_derivation_symbols:
                padding_seq = [self.padding] * (self.max_derivation_symbols - len(rhs_symbol))
                rhs_symbol.extend(padding_seq)
                rhs_exact_token.extend(padding_seq)
                parental_growth.extend(padding_seq)
                mask.extend(padding_seq)

            for k, l in zip(self.output_keys[:-1], (rhs_symbol, parental_growth, rhs_exact_token, mask)):
                left_most_derivation[k].extend(l)

        target_tokens = list(filter(lambda x: x not in (self.padding,), left_most_derivation[self.output_keys[-3]]))

        output = dict()
        for k, l in left_most_derivation.items():
            output[k] = torch.tensor(l)

        output[self.output_keys[-1]] = torch.tensor(target_tokens)
        return output

