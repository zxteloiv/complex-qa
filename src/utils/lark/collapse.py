from typing import List, Dict, Union, Generator, Tuple, Any
import lark
from .subtrees import EPS_TOK
TREE, TOKEN = lark.Tree, lark.Token


def collapse_trees(trees: List[TREE], idiom: TREE) -> None:
    for t in trees:
        collapse_tree(t, idiom)


def collapse_tree(tree: TREE, idiom: TREE) -> None:
    """traverse and collapse the tree matched by the idiom on-the-fly"""
    stack = [tree]
    while len(stack) > 0:
        node = stack.pop()
        # check if the node match the idiom, when matched, the collapse position is
        # the index of the dep-1 node to collapse for the idiom
        collapse_pos = subtree_matches_idiom(node, idiom)
        if collapse_pos >= 0:
            grandchildren = node.children[collapse_pos].children
            node.children = node.children[:collapse_pos] + grandchildren + node.children[collapse_pos + 1:]
        for n in reversed([c for c in node.children if isinstance(c, TREE)]):
            stack.append(n)


def subtree_matches_idiom(node: TREE, idiom: TREE) -> int:
    def _symbol_comp(n1: Union[TOKEN, TREE], n2: Union[TOKEN, TREE]) -> bool:
        if type(n1) != type(n2):
            return False
        if isinstance(n1, TOKEN):
            if n1.type != n2.type:  # for token only 1 criteria: two type must match
                return False
        else:
            if n1.data != n2.data:
                return False
        return True

    def _single_layer_comp(n1: TREE, n2: TREE) -> bool:
        if not _symbol_comp(n1, n2):
            return False
        if not isinstance(n1, TREE):
            return False
        if len(n1.children) > 0 and len(n1.children) != len(n2.children):
            return False
        elif len(n1.children) == 0 and not (n2.children == 1 and _symbol_comp(n2.children[0], EPS_TOK)):
            return False

        for c1, c2 in zip(n1.children, n2.children):
            if not _symbol_comp(c1, c2):
                return False
        return True

    if not _single_layer_comp(node, idiom):
        return -1
    i, child = next(filter(lambda c: isinstance(c[1], TREE) and len(c[1].children) > 0, enumerate(idiom.children)))
    if not _symbol_comp(node.children[i], child):
        return -1
    return i

