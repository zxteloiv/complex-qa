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
        self.padding = padding
        self.max_derivation_symbols = max_derivation_symbols
        self.grammar_token_generation = grammar_token_generation
        self.output_keys = output_keys or (
            'tree_nodes',           # (batch, n_d), the tree nodes, i.e., all the expanded lhs
            'node_parents',         # (batch, n_d),
            'expansion_frontiers',  # (batch, n_d, max_runtime_stack_size),
            'derivations',          # (batch, n_d, max_seq), the gold derivations, actually num_derivations = num_lhs
            'exact_tokens',         # (batch, n_d, max_seq),
            'target_tokens',        # (batch, tgt_len),
        )

    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        output = dict()
        for k in self.output_keys:
            tensor_list = tensors_by_keys[k]
            output[k] = self._nested_list_numbers_to_tensors(tensor_list)
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
        # to get the symbol_id and the exact_token id from a token string
        s_id = lambda t: self.vocab.get_token_index(t, self.namespaces[0])
        et_id = lambda t: self.vocab.get_token_index(t, self.namespaces[1])

        tree_nodes = []
        node_parent = []
        frontiers = []
        derivations = []
        exact_tokens = []
        target_tokens = []

        # pre-assign an ID, otherwise the system won't work
        # IDs are allocated in the left-most derivation order
        _id = 0
        def _assign_id_to_tree(t: Tree):
            nonlocal _id
            t.id = _id
            _id += 1
            for c in t.children:
                if isinstance(c, Tree):
                    _assign_id_to_tree(c)

        _assign_id_to_tree(tree)

        # traverse again the tree
        op_stack: List[Tuple[int, Tree]] = [(0, tree)]
        while len(op_stack) > 0:
            stack_node_ids = [t.id for _, t in op_stack]

            parent_id, node = op_stack.pop()
            tree_nodes.append(s_id(node.data))
            node_parent.append(parent_id)

            children: List[Union[Tree, Token]] = node.children
            _expansion = [s_id(START_SYMBOL)] + [s_id(s.data) if isinstance(s, Tree) else s_id(s.type) for s in children]
            _etokens = [et_id(START_SYMBOL)] + [et_id(s.value.lower()) if isinstance(s, Token) else self.padding for s in children]
            derivations.append(_expansion)
            exact_tokens.append(_etokens)
            target_tokens.extend(list(filter(lambda t: t != self.padding, _etokens)))
            frontiers.append(stack_node_ids)

            for c in reversed(children):
                if isinstance(c, Tree):
                    op_stack.append((node.id, c))

        output = dict(zip(self.output_keys, (
            tree_nodes,
            node_parent,
            frontiers,
            derivations,
            exact_tokens,
            target_tokens
        )))

        return output

    @staticmethod
    def _nested_list_numbers_to_tensors(nested: list, padding=0, example=None):
        """Turn a list of list of list of list ... of integers to a tensor with the given padding"""
        ndim_max = defaultdict(lambda: 0)

        def _count_nested_max(nested, depth):
            if not isinstance(nested, list):
                return

            ndim_max[depth] = max(ndim_max[depth], len(nested))
            for x in nested:
                _count_nested_max(x, depth + 1)

        _count_nested_max(nested, 0)
        ndim_max = [ndim_max[d] for d in sorted(ndim_max.keys())]

        def _get_padding_at_depth(depth):
            size = ndim_max[depth:]
            lump = padding
            for i in reversed(size):
                lump = [lump] * i
            return lump

        def _pad_nested(nested, depth):
            if not isinstance(nested, list):
                return nested

            if len(nested) < ndim_max[depth]:
                nested = nested + [_get_padding_at_depth(depth + 1)] * (ndim_max[depth] - len(nested))

            return [_pad_nested(x, depth + 1) for x in nested]

        full_fledged = _pad_nested(nested, 0)
        dev = dtype = None
        if example is not None:
            dev = example.device
            dtype = example.dtype

        return torch.tensor(full_fledged, device=dev, dtype=dtype)


