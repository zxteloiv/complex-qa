import math
import numbers
import os.path
import sys
import pandas as pd
from scipy.stats import norm
from collections import OrderedDict
import json


def main():
    for i, f in enumerate(sys.argv[1:]):
        file_proc(f, i == 0)


def file_proc(filename, print_header: bool = True):
    from utils.text_tool import grep, scan, split

    data = grep(open(filename).readlines(), r'Metrics:|:distance:')
    groups = split(data, 'Metrics: ')

    train, dev, test = [[float(n) for n in grep(grep(g, ':distance:'), r'\d+\.\d+', only_matched_text=True)]
                        for g in groups]

    param_p = norm.fit(train)
    param_q = norm.fit(test)
    divergence = kl_for_norms(*param_p, *param_q) + kl_for_norms(*param_q, *param_p)
    _, ds, method, seed = filename.split('.')
    out = OrderedDict(filename=filename, div=divergence, ds=ds, method=method)
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


if __name__ == '__main__':
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC', os.path.dirname(__file__)))
    main()
