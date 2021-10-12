from typing import List, Dict, Union, Generator, Tuple, Any
import logging
from os.path import join
import dill
from trialbot.utils.root_finder import find_root
import lark
import utils.cfg as cfg
from datetime import datetime as dt
from random import sample
from .eval import step_evaluation
from utils.lark.compact_hash import compact_hash
from utils.lark.stat import RuleCollector
from utils.lark.subtrees import generate_dep2_tree
from utils.lark.collapse import collapse_trees
from utils.lark.restore_cfg import restore_grammar, export_grammar
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
        self.collector = RuleCollector()
        self.max_mining_steps = max_mining_steps
        self.freq_lower_bound = freq_lower_bound
        self.prefix = data_prefix
        self.approx = sample_percentage
        self.retain_recursion = retain_recursion

    def mine(self):
        k = freq = 0
        for k in range(self.max_mining_steps):
            try:
                stat_train = self.collector.run_for_statistics(self.trees, k)
                logging.debug(f"finished calculating the training cfg .. {dt.now().strftime('%H%M%S')}")
                stat_eval = self.collector.run_for_statistics(self.eval_trees, k)
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
                collapse_trees(self.trees, d2tree)
                logging.debug(f"end of collapsing training data.. {dt.now().strftime('%H%M%S')}")
                collapse_trees(self.eval_trees, d2tree)
                logging.debug(f"end of collapsing dev data.. {dt.now().strftime('%H%M%S')}")

            except KeyboardInterrupt:
                logging.info("Keyboard interrupt detected. Stop mining and returns statistics")
                break
        dill.dump(self, open(self.prefix + f"{self.name}.{k}.miner_state", "wb"))

    def export_kth_rules(self, k, lex_in, start: cfg.NonTerminal,
                         export_terminals: bool = False, excluded_terminals = None,
                         remove_eps: bool = True,
                         remove_useless: bool = True,):
        g, terminal_vals = restore_grammar(self.stat_by_iter[k][0]['rule_dist'], self.collector)

        # we do not remove the unit rules since the extraction algorithm will remove them during itertaions
        g_txt = export_grammar(g, start,
                               join(find_root(), 'src', 'statics', 'grammar', lex_in),
                               terminal_vals if export_terminals else None,
                               excluded_terminals, remove_eps, remove_useless,
                               remove_unit_rules=False)

        name_pref = self.name
        if not remove_eps:
            name_pref += '.retain_eps'
        if not remove_useless:
            name_pref += '.retain_useless'

        with open(self.prefix + f"{name_pref}.{k}.lark", 'w') as fout:
            fout.write(g_txt)

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
            for d2t in generate_dep2_tree(t, self.retain_recursion):
                h = compact_hash(d2t)
                if h in freq_table:
                    freq_table[h]["freq"] += 1
                else:
                    freq_table[h] = {"freq": 1, "tree": d2t}

        return freq_table

