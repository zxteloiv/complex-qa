from typing import Literal, List, Dict, Union, Generator, Tuple, Any, Optional, Mapping, Callable, Set
import sys
import os
from os.path import join
sys.path.insert(0, join('..', '..'))
import pickle
from utils.root_finder import find_root
import lark
TREE, TOKEN = lark.Tree, lark.Token
import logging
logging.basicConfig(level=logging.INFO)
from itertools import chain
from collections import Counter, OrderedDict, defaultdict
import numpy as np
import math
import io
import utils.cfg as cfg
from datetime import datetime as dt
from random import sample

from datasets.comp_gen_bundle import install_sql_datasets, Registry
install_sql_datasets()

def print_dataset_statistics():
    print(f"all_names: {' '.join(Registry._datasets.keys())}")
    for name in filter(lambda s: 'cg' in s, Registry._datasets.keys()):
        train, dev, test = Registry.get_dataset(name)
        print(f"{name}: train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

def compact_hash(t: Union[TREE, TOKEN]):
    if isinstance(t, TOKEN):
        return t.type

    lhs = t.data
    # only the categories of terminals is considered, the values are not used
    rhs = ' '.join(compact_hash(c) for c in t.children)
    return f"{lhs}: ({rhs})"

def step_evaluation(train_stat: dict, dev_stat: dict, rule_to_id: dict) -> Dict[str, Dict[str, float]]:
    stat = OrderedDict()
    # grammar count
    logging.debug(f"analyze grammar count .. {dt.now().strftime('%H%M%S')}")
    train_grammar_size = train_stat['grammar_size']
    stat["train_grammar_size"] = {"size": train_grammar_size}

    # how the grammar rules themselves are used (distribution of rule count), other rules not considered
    logging.debug(f"analyze rule freq distribution .. {dt.now().strftime('%H%M%S')}")
    train_counter: Counter = train_stat['rule_dist']
    train_rule_stats = describe(list(train_counter.values()))
    stat["train_rule_stats"] = train_rule_stats

    # how the trees are formed by the grammar, check tree height, numbers of grammar rules and distinct rules
    logging.debug(f"analyze training tree distribution .. {dt.now().strftime('%H%M%S')}")
    train_heights_stat = describe([x['height'] for x in train_stat['trees_stat']])
    train_rule_num_stat = describe([x['rule_num'] for x in train_stat['trees_stat']])
    train_distinct_rule_num_stat = describe([x['distinct_rule_num'] for x in train_stat['trees_stat']])
    stat["train_heights_stat"] = train_heights_stat
    stat["train_tree_rule_num_stat"] = train_rule_num_stat
    stat["train_distinct_rule_num_stat"] = train_distinct_rule_num_stat
    logging.debug(f"analyze dev tree distribution .. {dt.now().strftime('%H%M%S')}")
    dev_heights_stat = describe([x['height'] for x in dev_stat['trees_stat']])
    dev_rule_num_stat = describe([x['rule_num'] for x in dev_stat['trees_stat']])
    dev_distinct_rule_num_stat = describe([x['distinct_rule_num'] for x in dev_stat['trees_stat']])
    stat["dev_heights_stat"] = dev_heights_stat
    stat["dev_tree_rule_num_stat"] = dev_rule_num_stat
    stat["dev_distinct_rule_num_stat"] = dev_distinct_rule_num_stat

    # data likelihoods with the probability estimated from the data
    logging.debug(f"analyze likelihood with base prob .. {dt.now().strftime('%H%M%S')}")
    # stat["base_train_ll_rec"] = dataset_stat(train_counter, [x['rule_count'] for x in train_stat['trees_stat']], 'base', rule_to_id)
    stat["base_dev_ll_rec"] = dataset_stat(train_counter, [x['rule_count'] for x in dev_stat['trees_stat']], 'base', rule_to_id)
    logging.debug(f"analyze likelihood with full prob .. {dt.now().strftime('%H%M%S')}")
    # stat["full_train_ll_rec"] = dataset_stat(train_counter, [x['rule_count'] for x in train_stat['trees_stat']], 'full', rule_to_id)
    stat["full_dev_ll_rec"] = dataset_stat(train_counter, [x['rule_count'] for x in dev_stat['trees_stat']], 'full', rule_to_id)

    # how close the two MLE estimated measures are
    logging.debug(f"analyze kl divergence .. {dt.now().strftime('%H%M%S')}")
    kl_stats = kl_divergence(train_counter, dev_stat['rule_dist'])
    stat["kl_stats"] = kl_stats
    return stat

def kl_divergence(c1: Counter, c2: Counter):
    rset1, rset2 = set(c1.keys()), set(c2.keys())
    common_rules = rset1.intersection(rset2)
    union_rules = rset1.union(rset2)
    p1_rate = len(common_rules) / len(rset1) if len(rset1) > 0 else 0
    p2_rate = len(common_rules) / len(rset2) if len(rset2) > 0 else 0
    common_rate = len(common_rules) / len(union_rules) if len(union_rules) > 0 else 0

    kl_p_q = kl_q_p = 0
    total_p = sum([c1[(nt, rule_id)] for (nt, rule_id) in common_rules if (nt, rule_id) in c1])
    total_q = sum([c2[(nt, rule_id)] for (nt, rule_id) in common_rules if (nt, rule_id) in c2])
    for (nt, rule_id) in common_rules:
        p = rule_density(c1, rule_id, nt, 'base', total_p)
        q = rule_density(c2, rule_id, nt, 'base', total_q)
        kl_p_q += p * math.log(p / q)
        kl_q_p += q * math.log(q / p)

    smooth_p_q = smooth_q_p = 0
    for (nt, rule_id) in union_rules:
        p = rule_density(c1, rule_id, nt, 'oov', total_p)
        q = rule_density(c2, rule_id, nt, 'oov', total_q)
        smooth_p_q += p * math.log(p / q)
        smooth_q_p += q * math.log(q / p)

    return {"p1_coverage": p1_rate, "p2_coverage": p2_rate, "union_coverage": common_rate,
            "base_p_q": kl_p_q, "base_q_p": kl_q_p, "smooth_p_q": smooth_p_q, "smooth_q_p": smooth_q_p}

def rule_density(counter: Counter, rule_id: int, nt: str,
                 support: Literal['base', 'oov', 'full'],
                 total: int = None,
                 complete_rules: set = None,
                 ):
    total = total or sum(counter.values())
    rule_count = counter[(nt, rule_id)] if (nt, rule_id) in counter else 0

    if support == 'base':
        return rule_count / total
    elif support == 'oov':
        return rule_count / (total + 1) if rule_count > 0 else 1 / (total + 1)
    else:
        complete_rule_num = sum([1 for (nt, rule_id) in complete_rules if (nt, rule_id) not in counter])
        return rule_count / (total + complete_rule_num) if rule_count > 0 else 1 / (total + complete_rule_num)

def dataset_stat(counter: Counter, tree_rule_counts: List[Counter],
                 support: Literal['base', 'oov', 'full'] = 'base',
                 complete_rules: dict = None) -> Dict[str, float]:
    complete_nt_num = None
    if support == 'full':
        complete_nt_num = len(set([v['root'] for v in complete_rules.values() if v['born_in'] == 0]))

    success = []
    lls = []
    for tree_count in tree_rule_counts:
        ll_buffer = []
        for (nt, rule_id), freq in tree_count.items():
            density = cond_rule_density(counter, rule_id, nt, support, complete_nt_num)
            if density == 0:
                ll_buffer = None
                break
            ll_buffer.append(math.log(density) * freq)

        success.append(ll_buffer is not None)
        if ll_buffer is not None:
            lls.append(sum(ll_buffer))
    mean_ll = sum(lls) / len(lls) if len(lls) > 0 else 'nan'
    recall = sum(success) * 1. / len(success)
    return {"mean_likelihood": mean_ll, "recall": recall}

def cond_rule_density(counter: Counter, rule_id: int, nt: str,
                      support: Literal['base', 'oov', 'full'] = 'base',
                      complete_nt_num: Optional[int] = None,  # only required for the "full" support
                      ):
    """
    Return the conditional density of the rule under the root nt.
    The density is an MLE estimator from the data (calculated from the counts directly).
    However based on the different support set,
    some unknown rules could have 0 probability or a small smoothing value.
    """
    assert support != 'full' or complete_nt_num is not None
    nt_sum = sum([freq for (root, rule_id), freq in counter.items() if root == nt], start=0)
    rule_count = counter[(nt, rule_id)] if (nt, rule_id) in counter else 0
    total = sum(counter.values())
    if support == 'base':
        return rule_count / nt_sum if nt_sum > 0 else 0  # likelihood could be 0
    else:  # "oov" and "full"
        # We reserved 1 count for unseen rules given every known root, and 1 count for unseen roots.
        # it seems we are smoothing the P(root) P(rule|root) separately.
        # With this trick, no tree will receive 0 likelihood.
        # When the complete nt set is smoothed, i.e., using the full-nt rather than the oov setting,
        # the densities will be significantly different
        # if too many non-terminals are disappeared during the collapse operations.
        nt_num = len(set(root for root, _ in counter.keys())) + 1 if support != 'full' else complete_nt_num
        if nt_sum > 0:
            return rule_count / (nt_sum + 1) if rule_count > 0 else 1 / (nt_sum + 1)
        else:
            return 1 / (total + nt_num)

def describe(vals: list):
    a = np.array(vals)
    # quantiles = np.quantile(a, [0., .25, .5, .75, 1.])
    mean = np.mean(a)
    std = np.std(a)
    median = np.median(a)
    # return dict(zip(("mean", "std", "min", ".25", "median", ".75", "max"), (mean, std, *quantiles)))
    return dict(zip(("mean", "std", "median"), (mean, std, median)))


class GreedyIdiomMiner:
    EPS_RHS = [lark.Token('%%EPS%%', '%%EPS%%')]
    def __init__(self, trees: List[TREE],
                 eval_trees: List[TREE],
                 name: str = None,
                 max_mining_steps: int = 1000,
                 freq_lower_bound: int = 0,
                 data_prefix: str = "",
                 sample_percentage: float = 1,
                 ):
        self.stat_by_iter = []
        self.idioms = dict()
        self.name = name or "miner"
        self.trees = trees
        self.eval_trees = eval_trees
        self.rule_to_id = dict()
        self.max_mining_steps = max_mining_steps
        self.freq_lower_bound = freq_lower_bound
        self.prefix = data_prefix
        self.approx = sample_percentage

    def assign_rule_id(self, subtree: TREE, iteration_num: int):
        hash = compact_hash(subtree)
        if hash in self.rule_to_id:
            return self.rule_to_id[hash]['id']
        else:
            new_id = len(self.rule_to_id)
            self.rule_to_id[hash] = {'id': new_id, 'born_in': iteration_num, 'root': subtree.data, 'tree': subtree}
            return new_id

    def mine(self):
        k = freq = 0
        for k in range(self.max_mining_steps):
            try:
                stat_train = self.run_for_statistics(k, self.trees)
                logging.debug(f"finished calculating the training cfg .. {dt.now().strftime('%H%M%S')}")
                stat_eval = self.run_for_statistics(k, self.eval_trees)
                self.stat_by_iter.append((stat_train, stat_eval))
                logging.debug(f"start to mine the most frequent idiom.. {dt.now().strftime('%H%M%S')}")

                if self.approx < 1:
                    if k > 0 and freq < len(self.trees) * self.approx:
                        self.approx += .1
                    sample_num = int(len(self.trees) * self.approx)
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
        pickle.dump(self, open(self.prefix + f"{self.name}.{k}.miner_state", "wb"))

    def run_for_statistics(self, k: int, trees: List[TREE]):
        # tree height
        rule_counter = Counter()    # (rule, nt) -> int, count for rules, and save the NT for likelihood use
        tree_stats = []
        for t in trees:
            tree_rule_counter = Counter()
            rule_num = 0
            for nt, rule in self.generate_dep1_tree(t):
                rule_id = self.assign_rule_id(rule, k)
                rule_counter[(nt, rule_id)] += 1
                tree_rule_counter[(nt, rule_id)] += 1
                rule_num += 1

            tree_stats.append({
                "rule_num": rule_num,   # equivalent to sum of all frequencies from the tree_rule_counter
                "distinct_rule_num": len(tree_rule_counter),
                "height": self.get_tree_height(t),
                "rule_count": tree_rule_counter,
            })
        return {
            "rule_dist": rule_counter,
            "grammar_size": len(rule_counter),
            "trees_stat": tree_stats,
        }

    def export_kth_rules(self, k, lex_in):
        g = self._restore_grammar(self.stat_by_iter[k][0]['rule_dist'])
        # we do not remove the unit rules since the extraction algorithm will remove them during itertaions
        g = cfg.remove_eps_rules(g)
        g = cfg.remove_useless_rules(g, cfg.NonTerminal('parse'))

        grammar_text = io.StringIO()
        print(open(join(find_root(), 'src', 'statics', 'grammar', lex_in)).read(), file=grammar_text)
        for lhs, rhs_list in g.items():
            print(f"{lhs.name}: " +
                  ('\n' + (' ' * len(lhs.name)  )+ '| ').join(' '.join(t.name for t in rhs) for rhs in rhs_list),
                  file=grammar_text)

        with open(self.prefix + f"{self.name}.{k}.lark", 'w') as fout:
            fout.write(grammar_text.getvalue())

    def _restore_grammar(self, counter: Counter) -> cfg.T_CFG:
        rule_lookup_table: Dict[int, TREE] = dict((r['id'], r['tree']) for hash, r in self.rule_to_id.items())
        g: cfg.T_CFG = defaultdict(list)
        def _transform(children: List[Union[TREE, TOKEN]]) -> list:
            rhs = []
            for c in children:
                if isinstance(c, TREE):
                    rhs.append(cfg.NonTerminal(c.data))
                else:
                    rhs.append(cfg.Terminal(c.type))
            return rhs
        for (nt, rid), count in counter.items():
            if count == 0:
                continue
            tree = rule_lookup_table[rid]
            g[cfg.NonTerminal(tree.data)].append(_transform(tree.children))
        return g

    def evaluation(self):
        stats: List[Dict[str, Dict[str, float]]] = []
        for k, (stat_train, stat_dev) in enumerate(self.stat_by_iter):
            logging.debug(f"start evaluation for the {k}th iteration .. {dt.now().strftime('%H%M%S')}")
            step_stats = step_evaluation(stat_train, stat_dev, self.rule_to_id)
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

    @classmethod
    def get_tree_height(cls, tree: TREE) -> int:
        if len(tree.children) == 0:
            return 1    # for any frontier NT node, assume there's an epsilon transition and its length 1

        # the root is defined as zero-height, and thus the height of terminal leaves are starting from 0
        return 1 + max(0 if isinstance(c, TOKEN) else cls.get_tree_height(c) for c in tree.children)

    @classmethod
    def get_d2tree_freq_table(cls, trees: List[TREE]) -> Dict[str, Dict[str, Any]]:
        freq_table = dict()
        for t in trees:
            for d2t in cls.generate_dep2_tree(t):
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
            elif len(n1.children) == 0 and not (n2.children == 1 and _symbol_comp(n2.children[0], cls.EPS_RHS[0])):
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

    @classmethod
    def generate_dep1_tree(cls, t: TREE) -> Generator[Tuple[str, TREE], None, None]:
        for st in t.iter_subtrees_topdown():
            dep1tree = lark.Tree(data=st.data, children=[
                # direct children (depth 1) copy assignments
                lark.Token(c.type, c.value) if isinstance(c, TOKEN)
                else lark.Tree(data=c.data, children=[])
                for c in st.children
            ] if len(st.children) > 0 else cls.EPS_RHS) # empty rule (A -> epsilon) must also be included
            yield st.data, dep1tree

    @classmethod
    def generate_dep2_tree(cls, t: TREE) -> Generator[TREE, None, None]:
        for st in t.iter_subtrees_topdown():
            # ignore the root when no grandchildren are available
            if len(st.children) == 0 or all(isinstance(c, TOKEN) for c in st.children):
                continue

            for i, child in enumerate(st.children):
                if isinstance(child, TOKEN):
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
                ] if len(child.children) > 0 else cls.EPS_RHS

                yield dep2tree

def sql_data_mining(prefix=""):
    names = [n for n in os.listdir(prefix) if n.endswith('parse.pkl')]
    for name in names:
        logging.info(f"================== {name} ====================")
        trees = pickle.load(open(prefix + name, 'rb'))
        miner = GreedyIdiomMiner(trees[0], trees[1], name[:name.index('.pkl')], data_prefix=prefix, freq_lower_bound=1)
        miner.mine()
        miner.evaluation()
        lex_file = 'SQLite.lark.lex-in' if 'sqlite' in name.lower() else 'MySQL.lark.lex-in'
        for i in range(0, len(miner.stat_by_iter), 10):
            miner.export_kth_rules(i, lex_file)

def cfq_dataset_mining():
    import datasets.cfq as cfq_data
    # train, dev, test = cfq_data.cfq_preparsed_treebase(join(cfq_data.CFQ_PATH, 'splits', 'mcd1.json'), conn=None)
    # train_tree = [obj['sparqlPatternModEntities_tree'] for obj in train]
    # dev_tree = [obj['sparqlPatternModEntities_tree'] for obj in dev]
    # miner = GreedyIdiomMiner(train_tree, dev_tree, 'cfq_mcd1', freq_lower_bound=3, data_prefix='run/', sample_percentage=.2)
    logging.debug(f"loading pickled cfq miner state .. {dt.now().strftime('%H%M%S')}")
    miner = pickle.load(open('run/cfq_mcd1.544.miner_state', 'rb'))
    # miner.mine()
    miner.evaluation()

def main():
    # sql part
    print_dataset_statistics()
    sql_data_mining(prefix='./run/')

    # sparql part
    # cfq_dataset_mining()

if __name__ == '__main__':
    main()
