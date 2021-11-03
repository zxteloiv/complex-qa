"""An ID Tree contains ID to each node, but discards the information of Lark"""
from typing import Union
import lark
from utils.tree import Tree, PreorderTraverse


def build_from_lark_tree(subtree_or_tok: Union[lark.Tree, lark.Token], add_eps_nodes: bool = False):
    if isinstance(subtree_or_tok, lark.Token):
        return Tree(subtree_or_tok.value, is_terminal=True)

    tree = Tree(subtree_or_tok.data, is_terminal=False, children=[
        Tree(Tree.EPS_TOK, is_terminal=True, children=[])
    ] if len(subtree_or_tok.children) == 0 and add_eps_nodes else [
        build_from_lark_tree(c, add_eps_nodes=add_eps_nodes) for c in subtree_or_tok.children
    ])
    return tree


if __name__ == '__main__':
    import os.path as osp, sys
    sys.path.insert(0, '../..')
    sparql_lark = osp.join('..', '..', 'statics', 'grammar', 'sparql_pattern.bnf.lark')
    sparql_parser = lark.Lark(open(sparql_lark), start="queryunit", keep_all_tokens=True,)
    sparql = r"""
        SELECT DISTINCT ?x0 WHERE {
        ?x0 P0 M1 .
        ?x0 P0 M2 .
        ?x0 a M0 .
        FILTER ( ?x0 != M1 ) .
        FILTER ( ?x0 != M2 )
        }
        """
    tree = sparql_parser.parse(sparql)
    id_tree = build_from_lark_tree(tree, add_eps_nodes=True).assign_node_id(PreorderTraverse())
    print(id_tree)

    # for n in PreorderTraverse()(id_tree):
    #     n: Tree
    #     print(n.node_id, ":", n.label, '-->', ' '.join(c.immediate_str() for c in n.children))

    for n in id_tree.iter_subtrees_topdown():
        n: Tree
        print(n.node_id, ":", n.label, '-->', ' '.join(c.immediate_str() for c in n.children))
