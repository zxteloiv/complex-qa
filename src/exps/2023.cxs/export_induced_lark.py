import logging
from typing import List, Union

import lark
from trialbot.utils.root_finder import find_root
import os.path as osp
import sys
import pickle
from collections import defaultdict, Counter

SRC_PATH = find_root('.SRC', osp.dirname(__file__))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from utils.cfg import T_CFG, restore_grammar_from_trees, simplify_grammar, NonTerminal
from utils.lark.restore_cfg import export_grammar
from utils.tree import Tree, PreOrderTraverse, PostOrderTraverse


logger = logging.getLogger(__name__)


def main(tree_file):
    data = pickle.load(open(tree_file, 'rb'))
    keys = ('x', 'y', 'x_id', 'y_id', 'y_tree')
    train_trees, dev_trees, test_trees = data[0]['y_tree'], data[1]['y_tree'], data[2]['y_tree']

    simplify_trees(train_trees)
    simplify_trees(dev_trees)
    simplify_trees(test_trees)

    g_txt = induce_grammar(train_trees)
    g_txt = "%import common.WS\n%ignore WS\n\n" + g_txt
    print(g_txt, file=open('induced-train.lark', 'w'))
    g_txt = induce_grammar(test_trees)
    g_txt = "%import common.WS\n%ignore WS\n\n" + g_txt
    print(g_txt, file=open('induced-test.lark', 'w'))
    g_txt = induce_grammar(dev_trees)
    g_txt = "%import common.WS\n%ignore WS\n\n" + g_txt
    print(g_txt, file=open('induced-dev.lark', 'w'))

    # parser = lark.Lark(g_txt, start='nt')
    # print('try parsing training set...')
    # a, c = try_reparse(parser, train_trees)
    # print(f'training set parsed...{a}/{c} = {a/c}')
    #
    # print('try parsing testing set...')
    # a, c = try_reparse(parser, test_trees)
    # print(f'testing set parsed...{a}/{c} = {a/c}')


def try_reparse(parser, trees):
    succ = []
    for i, t in enumerate(trees):
        print(f'the {i}th tree')
        y = ' '.join(n.label for n in PreOrderTraverse()(t) if n.is_terminal)
        try:
            t = parser.parse(y)
            succ.append(1)
        except:
            succ.append(0)
    return sum(succ), len(succ)


def induce_grammar(trees: List[Tree]):
    start = NonTerminal('start')
    g, terminals = restore_grammar_from_trees(trees)
    g_comp = prioritize_frequent_grammar_rules(g)
    g = simplify_grammar(g_comp, start)

    g_txt = export_grammar(g)
    return g_txt


def simplify_trees(trees: List[Tree]):
    def _find_redudant_node(tree: Tree):
        for node, parent, path in PostOrderTraverse(output_parent=True, output_path=True)(tree):
            if parent is not None and not node.is_terminal and len(node.children) <= 1:
                return node, parent, path
        return None

    for t in trees:
        redudant_node = _find_redudant_node(t)
        while redudant_node is not None:
            node, parent, path = redudant_node
            loc = path[-1]
            left_siblings = parent.children[:loc]
            right_siblings = parent.children[loc + 1:]
            parent.children = left_siblings + node.children + right_siblings
            redudant_node = _find_redudant_node(t)

    return trees


def prioritize_frequent_grammar_rules(g_occurrences: T_CFG):
    new_g = defaultdict(list)
    for nt, rhs_list in g_occurrences.items():
        counts = Counter(rhs_list)
        new_g[nt] = [rhs for rhs, _ in counts.most_common(len(counts))]
    return new_g


if __name__ == '__main__':
    main(sys.argv[1])

