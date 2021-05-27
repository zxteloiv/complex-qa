from typing import Literal, List, Dict, Union, Generator, Tuple, Any, Optional, Mapping, Callable, Set
import logging
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime as dt
import numpy as np
import math

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


