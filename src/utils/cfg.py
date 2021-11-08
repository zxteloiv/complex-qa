from typing import List, Dict, Tuple, Set, Callable, TypeVar, Generator, DefaultDict
from collections import defaultdict
__all__ = ['Symbol', 'NonTerminal', 'Terminal', 'T_CFG']
from utils.itertools import powerset
from utils.graph import dfs_walk, Graph
from utils.tree import Tree, PreorderTraverse


class Symbol:
    __slots__ = ('name',)

    is_terminal = NotImplemented

    def __init__(self, name: str):
        self.name: str = name

    def __eq__(self, other: 'Symbol'):
        assert isinstance(other, Symbol), other
        return self.is_terminal == other.is_terminal and self.name == other.name

    def __ne__(self, other: 'Symbol'):
        return not (self == other)

    def __hash__(self):
        return hash(self.name)  # implies different symbols (both T and NT) must have distinct names

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.name)

    fullrepr = property(__repr__)


class Terminal(Symbol):
    is_terminal = True


class NonTerminal(Symbol):
    is_terminal = False


T_CFG = Dict[NonTerminal, List[List[Symbol]]]


def restore_grammar_from_trees(trees: List[Tree],
                               export_terminal_values: bool = False
                               ) -> Tuple[T_CFG, DefaultDict[str, Set[str]]]:
    g = defaultdict(list)
    terminal_vals = defaultdict(set)
    for t in trees:
        for nt in t.iter_subtrees_topdown():
            lhs = NonTerminal(nt.label)
            rhs = [Terminal(c.label) if c.is_terminal else NonTerminal(c.label) for c in nt.children]
            g[lhs].append(rhs)
            if export_terminal_values:
                # export the terminal values when the terminals are in fact categories, whose single child
                # is the terminal value. this is true in real grammars to represent various terminal values
                # by a single category token. The values may be characterized by regular expressions or
                # wildcard characters defined in W3C EBNF standard.
                for c in nt.children:
                    # the only special case when there's a value under a terminal
                    if c.is_terminal and len(c.children) == 1:
                        terminal_vals[c.label].add(c.children[0].label)
    return g, terminal_vals


def simplify_grammar(g: T_CFG, start: NonTerminal = None) -> T_CFG:
    g = remove_eps_rules(g)
    g = remove_null_or_unit_rules(g)
    g = remove_useless_rules(g, start)
    return g


def remove_eps_rules(g: T_CFG) -> T_CFG:
    nullable_vars = set(nt for nt, rhs_list in g.items() if [Terminal(Tree.EPS_TOK)] in rhs_list)
    while True:
        nullable_vars_size = len(nullable_vars)
        for nt, rhs_list in g.items():
            if any(all(tok in nullable_vars for tok in rhs) for rhs in rhs_list):
                nullable_vars.add(nt)
        # repeat until no nullable vars are found
        if len(nullable_vars) == nullable_vars_size:
            break

    new_g = defaultdict(list)
    for nt, rhs_list in g.items():
        candidate_rhs_set: Set[Tuple[Symbol]] = set()
        for rhs in rhs_list:
            rhs_nv = set(rhs).intersection(nullable_vars)
            for nv_subset in powerset(rhs_nv):
                # new rules generated according to the powerset may happen to be the same, use a set for duplication
                new_rhs = list(filter(lambda tok: tok not in nv_subset, rhs))
                if len(new_rhs) > 0 and new_rhs != [Terminal(Tree.EPS_TOK)]:
                    candidate_rhs_set.add(tuple(new_rhs))

        for rhs in candidate_rhs_set:
            new_g[nt].append(list(rhs))
    return new_g


def remove_null_or_unit_rules(g: T_CFG) -> T_CFG:
    new_g = defaultdict(list)
    for nt, rhs_list in g.items():
        # add to new grammar the rules with rhs of 2 or more children
        new_g[nt].extend(rhs for rhs in rhs_list if len(rhs) > 1)
        # add to new grammar the rules with single terminal rhs
        new_g[nt].extend(rhs for rhs in rhs_list if len(rhs) == 1 and rhs[0].is_terminal)

    # build a dependency graph between unit-rhs rules (null, i.e. 0 children rhs is discarded)
    # in case for the looping cases like A -> B, B -> A
    depgraph: Graph[Symbol] = Graph()
    for nt, rhs_list in g.items():
        depgraph.add_v(nt)
        for rhs in filter(lambda rhs: len(rhs) == 1 and not rhs[0].is_terminal, rhs_list):
            depgraph.add_v(rhs[0])
            depgraph.add_e(nt, rhs[0])

    arrivals = {nt: list(dfs_walk(depgraph, nt, set())) for nt in depgraph.vertices}
    new_rules = defaultdict(list)
    for nt, nt_list in arrivals.items():
        for connected_nt in nt_list:
            new_rules[nt].extend(new_g[connected_nt])
    for nt, rhs_list in new_rules.items():
        new_g[nt].extend(rhs_list)
    return new_g


def remove_useless_rules(g: T_CFG, start: NonTerminal) -> T_CFG:
    if start is None:
        return g

    meaningful_symbols: Set[Symbol] = set()
    # --------- 1. remove the NonTerminals that will not derive any language -------------
    # add terminals
    meaningful_symbols.update(tok for rhs_list in g.values() for rhs in rhs_list for tok in rhs if tok.is_terminal)
    # add nonterminals until the set doesn't grow
    while True:
        old_size = len(meaningful_symbols)
        for nt, rhs_list in g.items():
            if any(all(tok.is_terminal or tok in meaningful_symbols for tok in rhs) for rhs in rhs_list):
                meaningful_symbols.add(nt)
        if len(meaningful_symbols) == old_size:
            break

    g1 = defaultdict(list)
    for nt, rhs_list in g.items():
        for rhs in rhs_list:
            if all(tok in meaningful_symbols for tok in rhs):
                g1[nt].append(rhs)

    # --------- 2. remove the NonTerminals not reachable from start -------------
    # remove only the nonterminals
    depgraph: Graph[Symbol] = Graph()
    for nt, rhs_list in g.items():
        depgraph.add_v(nt)
        for rhs in rhs_list:
            for tok in filter(lambda t: not t.is_terminal, rhs):
                depgraph.add_v(tok)
                depgraph.add_e(nt, tok)

    new_g = defaultdict(list)
    reachable: List[Symbol] = [start] + list(dfs_walk(depgraph, start, set()))
    for nt, rhs_list in g1.items():
        if nt not in reachable:
            continue
        for rhs in rhs_list:
            if all(tok.is_terminal or tok in reachable for tok in rhs):
                new_g[nt].append(rhs)

    return new_g

