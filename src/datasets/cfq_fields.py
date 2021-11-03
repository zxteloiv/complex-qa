from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Iterable, Union, DefaultDict
from itertools import product
from collections import defaultdict
import torch
from trialbot.data.field import Field
from trialbot.data.fields.seq_field import SeqField
import lark
from trialbot.data import START_SYMBOL, END_SYMBOL
from utils.preprocessing import nested_list_numbers_to_tensors
from utils.tree import Tree, PreorderTraverse
from utils.lark.id_tree import build_from_lark_tree


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


class TreeTraversalField(Field):
    def __init__(self, tree_key: str,
                 namespace: str,
                 output_keys: Optional[List[str]] = None,
                 padding: int = 0,
                 ):
        super().__init__()
        self.tree_key = tree_key
        self.ns = namespace or 'symbols'
        self.padding = padding
        self.output_keys: List[str] = output_keys or [
            'tree_nodes',           # (batch, n_d), the tree nodes, i.e., all the expanded lhs
            'node_parents',         # (batch, n_d),
            'expansion_frontiers',  # (batch, n_d, max_runtime_stack_size),
            'derivations',          # (batch, n_d, max_seq), the gold derivations, actually num_derivations = num_lhs
        ]

    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        output = dict()
        for k in self.output_keys:
            tensor_list = tensors_by_keys[k]
            output[k] = nested_list_numbers_to_tensors(tensor_list, padding=self.padding)
        return output

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        tree: lark.Tree = example.get(self.tree_key)
        if tree is not None:
            id_tree = build_from_lark_tree(tree, add_eps_nodes=True)
            for node in PreorderTraverse()(id_tree):
                yield from product([self.ns], [node.label, END_SYMBOL])

    def to_tensor(self, example) -> Mapping[str, Optional[list]]:
        tree = example.get(self.tree_key)
        if tree is None:
            return dict((k, None) for k in self.output_keys)

        # convert the lark tree to the decorated tree with IDs and eps rule expansions.
        id_tree = build_from_lark_tree(tree, add_eps_nodes=True)
        # pre-assign an ID, otherwise the system won't work
        # IDs are allocated in the left-most derivation order
        id_tree: Tree = id_tree.assign_node_id(PreorderTraverse())

        # to get the symbol_id and the exact_token id from a token string
        s_id = lambda t: self.vocab.get_token_index(t, self.ns)

        tree_nodes, node_parent, frontiers, derivations = [], [], [], []
        # traverse again the tree in the order of left-most derivation
        tree_node_id_set = set()
        op_stack: List[Tuple[int, Tree]] = [(0, id_tree)]
        while len(op_stack) > 0:
            parent_id, node = op_stack.pop()
            tree_nodes.append(s_id(node.label))
            node_parent.append(parent_id)

            _expansion = [s_id(c.label) for c in node.children] + [s_id(END_SYMBOL)]
            derivations.append(_expansion)

            tree_node_id_set.add(node.node_id)
            frontiers.append([x for x in tree_node_id_set if x not in node_parent])
            # the expanded nodes will all get attended over
            tree_node_id_set.update([c.node_id for c in node.children])

            for c in reversed(node.children):
                op_stack.append((node.node_id, c))

        frontiers[0].append(0)

        output = dict(zip(self.output_keys, (tree_nodes, node_parent, frontiers, derivations)))
        return output


class TutorBuilderField(Field):
    def __init__(self, tree_key: str, ns: str):
        super().__init__()
        self.tree_key = tree_key
        self.ns = ns

    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[DefaultDict]]) -> Mapping[str, DefaultDict]:
        grammar_map = defaultdict(set)
        for m in tensors_by_keys['grammar_map']:
            if m is None:
                continue
            for lhs, rhs_list in m.items():
                grammar_map[lhs].update(rhs_list)

        return {"grammar_map": grammar_map}

    def to_tensor(self, example) -> Mapping[str, Optional[DefaultDict[int, set]]]:
        tree = example.get(self.tree_key)
        if tree is None:
            return {"grammar_map": None}

        id_tree = build_from_lark_tree(tree, add_eps_nodes=True)
        grammar_map = defaultdict(set)
        for lhs, grammar in self._traverse_tree_for_derivations(id_tree):
            grammar_map[lhs].add(grammar)

        return {"grammar_map": grammar_map}

    def _traverse_tree_for_derivations(self, tree: Tree):
        s_id = lambda t: self.vocab.get_token_index(t, self.ns)
        for node in PreorderTraverse()(tree):
            children: List[Tree] = node.children
            lhs = s_id(node.label)
            rhs = [s_id(c.label) for c in children] + [s_id(END_SYMBOL)]
            p_growth = [0 if c.is_terminal else 1 for c in children] + [0]
            f_growth = [1] * (len(rhs) - 1) + [0]
            mask = [1] * len(rhs)
            grammar = [rhs, p_growth, f_growth, mask]
            grammar = tuple(map(tuple, grammar))
            yield lhs, grammar
