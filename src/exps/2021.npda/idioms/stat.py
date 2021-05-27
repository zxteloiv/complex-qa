from typing import Literal, List, Dict, Union, Generator, Tuple, Any, Optional, Mapping, Callable, Set
from collections import Counter, defaultdict
import lark
TREE, TOKEN = lark.Tree, lark.Token
from .compact_hash import compact_hash

EPS_RHS = [lark.Token('%%EPS%%', '%%EPS%%')]

class StatCollector:
    def __init__(self):
        self.rule_to_id = dict()

    def assign_rule_id(self, subtree: TREE, iteration_num: int):
        hash = compact_hash(subtree)
        if hash in self.rule_to_id:
            return self.rule_to_id[hash]['id']
        else:
            new_id = len(self.rule_to_id)
            self.rule_to_id[hash] = {'id': new_id, 'born_in': iteration_num, 'root': subtree.data, 'tree': subtree}
            return new_id

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

    @classmethod
    def generate_dep1_tree(cls, t: TREE) -> Generator[Tuple[str, TREE], None, None]:
        for st in t.iter_subtrees_topdown():
            dep1tree = lark.Tree(data=st.data, children=[
                # direct children (depth 1) copy assignments
                lark.Token(c.type, c.value) if isinstance(c, TOKEN)
                else lark.Tree(data=c.data, children=[])
                for c in st.children
            ] if len(st.children) > 0 else EPS_RHS) # empty rule (A -> epsilon) must also be included
            yield st.data, dep1tree

    @classmethod
    def get_tree_height(cls, tree: TREE) -> int:
        if len(tree.children) == 0:
            return 1    # for any frontier NT node, assume there's an epsilon transition and its length 1

        # the root is defined as zero-height, and thus the height of terminal leaves are starting from 0
        return 1 + max(0 if isinstance(c, TOKEN) else cls.get_tree_height(c) for c in tree.children)

