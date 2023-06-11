import math
import numbers
import os.path
import random
import statistics
import sys
from collections import OrderedDict
from typing import List
import numpy as np

import pandas as pd
from scipy.stats import norm

MAX_ITEMS = 2000
MAX_TRIALS = 10


def main():
    for i, f in enumerate(sys.argv[1:]):
        file_proc(f, i == 0)


def file_proc(filename, print_header: bool = True):
    from utils.text_tool import grep, split

    data = grep(open(filename).readlines(), r'Metrics:|:distance:')
    groups = split(data, 'Metrics: ')

    train, dev, test = [[float(n) for n in grep(grep(g, ':distance:'), r'\d+\.\d+', only_matched_text=True)]
                        for g in groups]

    param_p = norm.fit(train)
    param_q = norm.fit(test)
    divergence = kl_for_norms(*param_p, *param_q) + kl_for_norms(*param_q, *param_p)
    _, ds, method, seed = filename.split('.')
    method = 'seq' if method == 'moving_metric_flat' else 'prod' if method == 'moving_metric' else method
    out = OrderedDict(filename=filename, div=divergence, ds=ds, method=method, ot=ot_moving(train, test))
    for k, v in pd.Series(train).describe().items():
        out[f'train_{k}'] = v
    for k, v in pd.Series(test).describe().items():
        out[f'test_{k}'] = v

    if print_header:
        print('\t'.join(out.keys()))

    print('\t'.join('{0:.4f}'.format(v)
                    if isinstance(v, numbers.Number) else
                    '{0}'.format(v)
                    for v in out.values()))


def kl_for_norms(m1, s1, m2, s2):
    t1 = ((m2 - m1) / s2) ** 2
    t2 = (s1 / s2) ** 2
    return (t1 + t2 - math.log(t2) - 1) / 2


def ot_moving(xs: List[float], xt: List[float]):
    import ot

    def _moving_dist(small_xs: List[float], small_xt: List[float]):
        a = np.expand_dims(np.array(small_xs, dtype=np.float64), 1)
        b = np.expand_dims(np.array(small_xt, dtype=np.float64), 1)
        return ot.emd2([], [], ot.dist(a, b))

    if len(xs) <= MAX_ITEMS and len(xt) <= MAX_ITEMS:
        return _moving_dist(xs, xt)

    random.seed(2023)
    sample = random.sample

    emds = [_moving_dist(sample(xs, MAX_ITEMS) if len(xs) > MAX_ITEMS else xs,
                         sample(xt, MAX_ITEMS) if len(xt) > MAX_ITEMS else xt)
            for _ in range(MAX_TRIALS)]
    return statistics.fmean(emds)


if __name__ == '__main__':
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC', os.path.dirname(__file__)))
    main()
