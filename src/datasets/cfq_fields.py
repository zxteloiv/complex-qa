from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Iterable, Union, DefaultDict
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from .field import FieldAwareTranslator, Field
from .seq_field import SeqField
import lark
from trialbot.data import START_SYMBOL, END_SYMBOL
from utils.preprocessing import nested_list_numbers_to_tensors
_Tree, _Token = lark.Tree, lark.Token
from itertools import product

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
        if tree is not None:
            for subtree in tree.iter_subtrees_topdown():
                rule_str = self._get_rule_str(subtree)
                yield self.ns, rule_str

        if self.add_start_end_toks:
            yield from product([self.ns], [START_SYMBOL, END_SYMBOL])

    def to_tensor(self, example) -> Mapping[str, Optional[torch.Tensor]]:
        Tree, Token = lark.Tree, lark.Token
        tree: Tree = example.get(self.source_key)
        if tree is None:
            return {self.renamed_key: None}

        rule_id = [self.vocab.get_token_index(self._get_rule_str(st), namespace=self.ns) for st in tree.iter_subtrees_topdown()]

        if self.add_start_end_toks:
            start_id = self.vocab.get_token_index(START_SYMBOL, self.ns)
            end_id = self.vocab.get_token_index(END_SYMBOL, self.ns)
            rule_id = [start_id] + rule_id + [end_id]

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
            output[k] = nested_list_numbers_to_tensors(tensor_list)
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
        tree: _Tree = example.get(self.tree_key)
        assert tree is not None
        # to get the symbol_id and the exact_token id from a token string
        s_id = lambda t: self.vocab.get_token_index(t, self.namespaces[0])
        et_id = lambda t: self.vocab.get_token_index(t, self.namespaces[1])
        is_tree = lambda s: isinstance(s, _Tree)

        tree_nodes = []
        node_parent = []
        frontiers = []
        derivations = []
        exact_tokens = []
        target_tokens = []

        # pre-assign an ID, otherwise the system won't work
        # IDs are allocated in the left-most derivation order
        _id = 0
        def _assign_id_to_tree(t: _Tree):
            nonlocal _id
            t.id = _id
            _id += 1
            for c in t.children:
                if is_tree(c):
                    _assign_id_to_tree(c)

        _assign_id_to_tree(tree)

        tree_node_id_set = set()

        # traverse again the tree in the order of left-most derivation
        op_stack: List[Tuple[int, _Tree]] = [(0, tree)]
        while len(op_stack) > 0:
            parent_id, node = op_stack.pop()
            tree_nodes.append(s_id(node.data))
            node_parent.append(parent_id)

            children: List[Union[_Tree, _Token]] = node.children
            _expansion = [s_id(START_SYMBOL)] + [s_id(s.data) if is_tree(s) else s_id(s.type) for s in children]
            _etokens = [et_id(START_SYMBOL)] + [et_id(s.value.lower()) if not is_tree(s) else self.padding for s in children]
            derivations.append(_expansion)
            exact_tokens.append(_etokens)
            target_tokens.extend(list(filter(lambda t: t != self.padding, _etokens)))

            tree_node_id_set.add(node.id)
            frontiers.append([x for x in tree_node_id_set if x not in node_parent])
            # the expanded nodes will all get attended over
            tree_node_id_set.update([n.id for n in children if is_tree(n)])

            for c in reversed(children):
                if is_tree(c):
                    op_stack.append((node.id, c))

        frontiers[0].append(0)

        output = dict(zip(self.output_keys, (
            tree_nodes,
            node_parent,
            frontiers,
            derivations,
            exact_tokens,
            target_tokens
        )))

        return output

class TutorBuilderField(Field):
    def __init__(self, tree_key: str, ns: Tuple[str, str]):
        super().__init__()
        self.tree_key = tree_key
        self.ns = ns

    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[DefaultDict]]) -> Mapping[str, DefaultDict]:
        token_map = defaultdict(set)
        for m in tensors_by_keys['token_map']:
            for symbol, tokens in m.items():
                token_map[symbol].update(tokens)

        grammar_map = defaultdict(set)
        for m in tensors_by_keys['grammar_map']:
            for lhs, rhs_list in m.items():
                grammar_map[lhs].update(rhs_list)

        return {"token_map": token_map, "grammar_map": grammar_map}

    def to_tensor(self, example) -> Mapping[str, DefaultDict[int, list]]:
        tree: _Tree = example.get(self.tree_key)
        token_map = defaultdict(set)
        for k, v in self._generate_exact_token_mapping(tree):
            token_map[k].add(v)

        grammar_map = defaultdict(set)
        for lhs, grammar in self._traverse_tree_for_derivations(tree):
            grammar_map[lhs].add(grammar)

        return {"token_map": token_map, "grammar_map": grammar_map}

    def _generate_exact_token_mapping(self, tree: _Tree):
        s_id = lambda t: self.vocab.get_token_index(t, self.ns[0])
        et_id = lambda t: self.vocab.get_token_index(t, self.ns[1])
        yield s_id(START_SYMBOL), et_id(START_SYMBOL)
        for subtree in tree.iter_subtrees_topdown():
            for s in subtree.children:
                if isinstance(s, _Token):
                    yield s_id(s.type), et_id(s.value.lower())

    def _traverse_tree_for_derivations(self, tree: _Tree):
        is_tree = lambda s: isinstance(s, _Tree)
        s_id = lambda t: self.vocab.get_token_index(t, self.ns[0])
        for subtree in tree.iter_subtrees_topdown():
            children: List[Union[_Tree, _Token]] = subtree.children
            lhs = s_id(subtree.data)
            rhs = [s_id(START_SYMBOL)] + [s_id(s.data) if is_tree(s) else s_id(s.type) for s in children]
            p_growth = [0] + [1 if is_tree(s) else 0 for s in children]
            f_growth = [1] * (len(rhs) - 1) + [0]
            mask = [1] * len(rhs)
            grammar = [rhs, p_growth, f_growth, mask]
            grammar = tuple(map(tuple, grammar))
            yield lhs, grammar
