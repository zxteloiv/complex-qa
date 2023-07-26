import lark
import trialbot.utils.prepend_pythonpath    # noqa
import logging
import pickle
from datetime import datetime as dt
from collections import defaultdict, Counter
from collections.abc import Iterable

from utils.tree import Tree, PreOrderTraverse, PostOrderTraverse, InOrderTraverse
from utils.cfg import T_CFG, restore_grammar_from_trees, simplify_grammar, NonTerminal
from utils.lark.restore_cfg import export_grammar

logging.getLogger().setLevel(logging.INFO)


def timestr(fmt='%H:%M:%S'):
    return dt.now().strftime(fmt)


def main(files: list[str]):
    logging.info(f'start loading {files} @{timestr()}')
    data: list[list[Tree]] = [pickle.load(open(f, 'rb')) for f in files]
    logging.info(f'loaded trees from {files} @{timestr()}')

    train, dev, test = data

    g_txt = induce_grammar(train)
    g_txt = "%import common.WS\n%ignore WS\n\n" + g_txt
    g_filename = 'train-g.lark'
    print(g_txt, file=open(g_filename, 'w'))
    logging.info(f'save dump grammar to {g_filename} @{timestr()}')

    rule_set = build_rule_set(train)
    logging.info(f'built rule set @{timestr()}')

    # logging.info(f'checking dev data @{timestr()}')
    check_ds(dev, rule_set)

    logging.info(f'checking testing data @{timestr()}')
    check_ds(test, rule_set)


def build_rule_set(ts: Iterable[Tree]) -> set[str]:
    return set(n.immediate_str()
               for t in ts
               for n in t.iter_subtrees_topdown())


def check_ds(ts: Iterable[Tree], rule_set: set[str]):
    acc, num = 0, 0
    for i, tree in enumerate(ts):
        num += 1

        for n in PreOrderTraverse()(tree):
            n: Tree
            if not n.is_terminal:
                rule_str = n.immediate_str()
                if rule_str not in rule_set:
                    # logging.warning(f'unseen rule found: {rule_str}')
                    break
        else:
            acc += 1

    logging.info(f'succ {acc / num:.4f} with total {num}')
    return acc, num


def parse_ds(ts: Iterable[Tree], parser: lark.Lark):
    acc, num = 0, 0
    for i, tree in enumerate(ts):
        terms = terms_from_tree(tree)
        try:
            reconstructed = parser.parse(terms, start='nt')
            acc += 1
            logging.info(f'success for data {i}')
        except:
            logging.warning(f'failed for data {i} @{timestr()}')
        finally:
            num += 1

    logging.info(f'succ {acc / num:.4f} with total {num}')
    return acc, num


def terms_from_tree(tree: Tree) -> str:
    n: Tree
    return ' '.join(n.label for n in PreOrderTraverse()(tree) if n.is_terminal)


def induce_grammar(trees: list[Tree]):
    start = NonTerminal('nt')
    g, terminals = restore_grammar_from_trees(trees)
    g_comp = prioritize_frequent_grammar_rules(g)
    g = simplify_grammar(g_comp, start)

    g_txt = export_grammar(g)
    return g_txt


def prioritize_frequent_grammar_rules(g_occurrences: T_CFG):
    new_g = defaultdict(list)
    for nt, rhs_list in g_occurrences.items():
        counts = Counter(rhs_list)
        new_g[nt] = [rhs for rhs, _ in counts.most_common(len(counts))]
    return new_g


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

