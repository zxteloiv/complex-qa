from typing import Literal, List, Dict, Union, Generator, Tuple, Any, Optional, Mapping, Callable, Set
import logging
from os.path import join
import dill
from trialbot.utils.root_finder import find_root
import lark
from collections import Counter, defaultdict
import io
import utils.cfg as cfg
from datetime import datetime as dt
from random import sample
from .eval import step_evaluation
from .compact_hash import compact_hash
from .stat import EPS_RHS, StatCollector
TREE, TOKEN = lark.Tree, lark.Token


class GreedyIdiomMiner:
    def __init__(self, trees: List[TREE],
                 eval_trees: List[TREE],
                 name: str = None,
                 max_mining_steps: int = 1000,
                 freq_lower_bound: int = 0,
                 data_prefix: str = "",
                 sample_percentage: float = 1,
                 retain_recursion: bool = True,
                 ):
        self.stat_by_iter = []
        self.idioms = dict()
        self.name = name or "miner"
        self.trees = trees
        self.eval_trees = eval_trees
        self.collector = StatCollector()
        self.max_mining_steps = max_mining_steps
        self.freq_lower_bound = freq_lower_bound
        self.prefix = data_prefix
        self.approx = sample_percentage
        self.retain_recursion = retain_recursion

    def mine(self):
        k = freq = 0
        for k in range(self.max_mining_steps):
            try:
                stat_train = self.collector.run_for_statistics(k, self.trees)
                logging.debug(f"finished calculating the training cfg .. {dt.now().strftime('%H%M%S')}")
                stat_eval = self.collector.run_for_statistics(k, self.eval_trees)
                self.stat_by_iter.append((stat_train, stat_eval))
                logging.debug(f"start to mine the most frequent idiom.. {dt.now().strftime('%H%M%S')}")

                if self.approx < 1:
                    if k > 0 and freq < len(self.trees) * self.approx:
                        self.approx += .1
                    sample_num = min(int(len(self.trees) * self.approx), len(self.trees))
                    d2tree_freq_table = self.get_d2tree_freq_table(sample(self.trees, sample_num))
                else:
                    d2tree_freq_table = self.get_d2tree_freq_table(self.trees)

                if len(d2tree_freq_table) < 1: # all the parse is now depth 1
                    logging.info('no more depth-2 grammar available, quit and save now.')
                    break

                most_freq_d2t = max(d2tree_freq_table.values(), key=lambda x: x["freq"])
                d2tree, freq = most_freq_d2t['tree'], most_freq_d2t['freq']
                if freq <= self.freq_lower_bound: # all the parse is now depth 1
                    logging.info('depth-2 grammar rules are almost distinct, quit and save now.')
                    break

                self.idioms[compact_hash(d2tree)] = {"tree": d2tree, "born_in": k, "freq": freq}
                logging.info(f"idiom @ {k}: {freq} -> {compact_hash(d2tree)}")
                self.collapse_trees(self.trees, d2tree)
                logging.debug(f"end of collapsing training data.. {dt.now().strftime('%H%M%S')}")
                self.collapse_trees(self.eval_trees, d2tree)
                logging.debug(f"end of collapsing dev data.. {dt.now().strftime('%H%M%S')}")

            except KeyboardInterrupt:
                logging.info("Keyboard interrupt detected. Stop mining and returns statistics")
                break
        dill.dump(self, open(self.prefix + f"{self.name}.{k}.miner_state", "wb"))

    def export_kth_rules(self, k, lex_in, start: cfg.NonTerminal,
                         export_terminals: bool = False, excluded_terminals = None,
                         remove_eps: bool = True,
                         remove_useless: bool = True,):
        g, terminal_vals = self._restore_grammar(self.stat_by_iter[k][0]['rule_dist'])
        # we do not remove the unit rules since the extraction algorithm will remove them during itertaions
        if remove_eps:
            g = cfg.remove_eps_rules(g)
        if remove_useless:
            g = cfg.remove_useless_rules(g, start)

        grammar_text = io.StringIO()
        print(open(join(find_root(), 'src', 'statics', 'grammar', lex_in)).read(), file=grammar_text)
        for lhs, rhs_list in g.items():
            print(f"{lhs.name}: " + ('\n' + (' ' * len(lhs.name)  )+ '| ').join(
                ' '.join(f"P{t.name}" if t.name.startswith('_') else t.name for t in rhs)
                for rhs in rhs_list
            ), file=grammar_text)

        if export_terminals:
            for tok, vals in terminal_vals.items():
                if tok not in excluded_terminals and tok != EPS_RHS[0].type:
                    if tok.startswith('_'):
                        tok = 'P' + tok
                    print(f"{tok}: " + ('\n' + (' ' * len(tok)) + '| ').join(
                        f'"{val}"i' if any(letter in val.lower() for letter in 'abcdefghijklmnopqrstuvwxyz')
                        else f'"{val}"'
                        for val in vals
                    ), file=grammar_text)

        name_pref = self.name
        if not remove_eps:
            name_pref += '.retain_eps'
        if not remove_useless:
            name_pref += '.retain_useless'

        with open(self.prefix + f"{name_pref}.{k}.lark", 'w') as fout:
            fout.write(grammar_text.getvalue())

    def _restore_grammar(self, counter: Counter) -> Tuple[cfg.T_CFG, dict]:
        terminal_vals = defaultdict(set)
        rule_lookup_table: Dict[int, TREE] = dict((r['id'], r['tree']) for hash, r in self.collector.rule_to_id.items())
        g: cfg.T_CFG = defaultdict(list)
        def _transform(children: List[Union[TREE, TOKEN]]) -> list:
            rhs = []
            for c in children:
                if isinstance(c, TREE):
                    rhs.append(cfg.NonTerminal(c.data))
                else:
                    rhs.append(cfg.Terminal(c.type))
                    terminal_vals[c.type].add(c.value)
            return rhs
        for (nt, rid), count in counter.items():
            if count == 0:
                continue
            tree = rule_lookup_table[rid]
            g[cfg.NonTerminal(tree.data)].append(_transform(tree.children))
        return g, terminal_vals

    def evaluation(self):
        stats: List[Dict[str, Dict[str, float]]] = []
        for k, (stat_train, stat_dev) in enumerate(self.stat_by_iter):
            logging.debug(f"start evaluation for the {k}th iteration .. {dt.now().strftime('%H%M%S')}")
            step_stats = step_evaluation(stat_train, stat_dev, self.collector.rule_to_id)
            logging.debug(f"finish evaluation for the {k}th iteration .. {dt.now().strftime('%H%M%S')}")
            stats.append(step_stats)

        if len(stats)> 0:
            with open(self.prefix + f"{self.name}.grammar_stat.tsv", "w") as f:
                h1 = stats[0].keys()
                h1_size = [len(v.keys()) for v in stats[0].values()]
                for t, l in zip(h1, h1_size):
                    f.write(t)
                    while l > 0:
                        l -= 1
                        f.write('\t')
                f.write('\n')
                h2 = sum([list(k + '_' + h for h in v.keys()) for k, v in stats[0].items()], start=[])
                print("\t".join(h2), file=f)

                for l in stats:
                    val_list = sum([list(v.values()) for v in l.values()], start=[])
                    print(('%.4f\t' * len(val_list)).rstrip() % tuple(val_list), file=f)
                f.close()
        else:
            logging.warning("No stats found.")

        return stats

    def get_d2tree_freq_table(self, trees: List[TREE]) -> Dict[str, Dict[str, Any]]:
        freq_table = dict()
        for t in trees:
            for d2t in self.generate_dep2_tree(t):
                h = compact_hash(d2t)
                if h in freq_table:
                    freq_table[h]["freq"] += 1
                else:
                    freq_table[h] = {"freq": 1, "tree": d2t}

        return freq_table

    @classmethod
    def collapse_trees(cls, trees: List[TREE], idiom: TREE) -> None:
        for t in trees:
            cls.collapse_tree(t, idiom)

    @classmethod
    def collapse_tree(cls, tree: TREE, idiom: TREE) -> None:
        """traverse and collapse the tree matched by the idiom on-the-fly"""
        stack = [tree]
        while len(stack) > 0:
            node = stack.pop()
            # check if the node match the idiom, when matched, the collapse position is
            # the index of the dep-1 node to collapse for the idiom
            collapse_pos = cls.subtree_matches_idiom(node, idiom)
            if collapse_pos >= 0:
                grandchildren = node.children[collapse_pos].children
                node.children = node.children[:collapse_pos] + grandchildren + node.children[collapse_pos + 1:]
            for n in reversed([c for c in node.children if isinstance(c, TREE)]):
                stack.append(n)

    @classmethod
    def subtree_matches_idiom(cls, node: TREE, idiom: TREE) -> int:
        def _symbol_comp(n1: Union[TOKEN, TREE], n2: Union[TOKEN, TREE]) -> bool:
            if type(n1) != type(n2):
                return False
            if isinstance(n1, TOKEN):
                if n1.type != n2.type:    # for token only 1 criteria: two type must match
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
            elif len(n1.children) == 0 and not (n2.children == 1 and _symbol_comp(n2.children[0], EPS_RHS[0])):
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

    def generate_dep2_tree(self, t: TREE) -> Generator[TREE, None, None]:
        for st in t.iter_subtrees_topdown():
            # ignore the root when no grandchildren are available
            if len(st.children) == 0 or all(isinstance(c, TOKEN) for c in st.children):
                continue

            for i, child in enumerate(st.children):
                if isinstance(child, TOKEN):
                    continue

                if self.retain_recursion and child.data == st.data:
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

