from typing import List, Generator, Tuple
from collections import Counter
import lark

TREE, TOKEN = lark.Tree, lark.Token
EPS_TOK = lark.Token('%%EPS%%', '%%EPS%%')
EPS_RHS = [EPS_TOK]


def generate_dep1_tree(t: TREE) -> Generator[Tuple[str, TREE], None, None]:
    for st in t.iter_subtrees_topdown():
        dep1tree = lark.Tree(data=st.data, children=[
            # direct children (depth 1) copy assignments
            lark.Token(c.type, c.value) if isinstance(c, TOKEN)
            else lark.Tree(data=c.data, children=[])
            for c in st.children
        ] if len(st.children) > 0 else EPS_RHS)  # empty rule (A -> epsilon) must also be included
        yield st.data, dep1tree


def generate_dep2_tree(t: TREE, retain_recursion: bool = False) -> Generator[TREE, None, None]:
    for st in t.iter_subtrees_topdown():
        # ignore the root when no grandchildren are available
        if len(st.children) == 0 or all(isinstance(c, TOKEN) for c in st.children):
            continue

        for i, child in enumerate(st.children):
            if isinstance(child, TOKEN):
                continue

            if retain_recursion and child.data == st.data:
                continue

            dep2tree = lark.Tree(data=st.data, children=[
                lark.Token(c.type, "") if isinstance(c, TOKEN)
                else lark.Tree(data=c.data, children=[]) # all the grandchildren is set empty first
                for c in st.children
            ])

            # only expand the i-th grandchildren
            dep2tree.children[i].children = [
                lark.Token(grandchild.type, "") if isinstance(grandchild, TOKEN)
                else lark.Tree(grandchild.data, children=[])    # the grand-grand-children is always empty
                for grandchild in child.children
            ] if len(child.children) > 0 else EPS_RHS

            yield dep2tree

