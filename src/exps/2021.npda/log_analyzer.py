# a utility to analyze the logs output from the
from typing import List
import re
import json
import pandas as pd

def process_tranx_log(data: List[str]):
    groups = split(scan(data, 'Epoch started'), '======*')
    metrics = []
    for g in groups[10:]:
        train_metrics = grep(grep(g, ':{.*}|Training Metrics'), '{.*}', only_matched_text=True)[-1]
        dev_metrics = grep(grep(g, 'Evaluation Metrics'), '{.*}', only_matched_text=True)[-1]
        test_metrics = grep(grep(g, 'Testing Metrics'), '{.*}', only_matched_text=True)[-1]
        g_m = tuple()
        g_m = g_m + tuple(json.loads(x)['ERR'] for x in (train_metrics, dev_metrics, test_metrics))
        g_m = g_m + tuple(json.loads(x)['ERR_TOPO'] for x in (train_metrics, dev_metrics, test_metrics))
        g_m = g_m + tuple(json.loads(x)['ERR_TOKN'] for x in (train_metrics, dev_metrics, test_metrics))
        metrics.append(g_m)

    print("\t".join(('epoch', 'train-err', 'dev-err', 'test-err',
                     'train-err-topo', 'dev-err-topo', 'test-err-topo',
                     'train-err-tokn', 'dev-err-tokn', 'test-err-tokn',
                     )))
    print("\n".join(("%d" + ('\t%.4f' * len(m))) % ((i,) + tuple(m)) for i, m in enumerate(metrics)))

    # hparam = grep(data, 'Hyperparamset.*', only_matched_text=True)
    # print(hparam[0])
    # f_eq = lambda x, y: abs(x - y) < 1e-8
    # least_train_err = min(m[0] for m in metrics)
    # least_dev_err = min(m[1] for m in metrics)
    # least_test_err = min(m[2] for m in metrics)
    # print(f"least train/dev/test: {least_train_err:.4}, {least_dev_err:.4}, {least_test_err:.4}")
    # metrics_by_least_train = [(i, m) for i, m in enumerate(metrics) if f_eq(m[0], least_train_err)]
    # metrics_by_least_dev = [(i, m) for i, m in enumerate(metrics) if f_eq(m[1], least_dev_err)]
    # metrics_by_least_test = [(i, m) for i, m in enumerate(metrics) if f_eq(m[2], least_test_err)]
    # print(metrics_by_least_train)
    # print(metrics_by_least_dev)
    # print(metrics_by_least_test)

def split(data: List[str], split_pat: str) -> List[List[str]]:
    split_pat = re.compile(split_pat)
    out: List[List[str]] = []
    group: List[str] = []
    for l in data:
        if split_pat.search(l) is not None and len(group) > 0:
            out.append(group)
            group = []
        else:
            group.append(l)
    if len(group) > 0:
        out.append(group)
    return out

def grep(data: List[str], pat: str, only_matched_text: bool = False) -> List[str]:
    pat = re.compile(pat)
    out = []
    for l in data:
        m = pat.search(l)
        if m is None:
            continue

        if only_matched_text:
            out.append(m.group())
        else:
            out.append(l)
    return out

def scan(data: List[str], start_pat: str = None, ending_pat: str = None, count: int = None, include_ending: bool = False) -> List[str]:
    assert count is None or isinstance(count, int) and count > 0
    start_pat = None if start_pat is None else re.compile(start_pat)
    ending_pat = None if ending_pat is None else re.compile(ending_pat)

    out = []
    i = 0
    while i < len(data):
        l = data[i]
        if start_pat is not None:
            m = start_pat.search(l)
            if m is None:
                i += 1
                continue

        # when either start_pat is None or start_pat is found, the start_pos is fixed at i
        if ending_pat is None and count is None:
            out.extend(data[i:])
            break

        if count is not None and count > 0:
            out.extend(data[i:i + count])
            i = i + count
            continue

        for j in range(i + 1, len(data)):
            ll = data[j]
            mm = ending_pat.search(ll)
            if mm is None:
                continue

            end_pos = j + 1 if include_ending else j
            out.extend(data[i:end_pos])
            # the ending is matched no matter included or not, so the next start pos will be the next line: j + 1
            i = j + 1
            break   # only the nearest ending pattern is required
        else:
            # if the ending pattern is never found after the text is exhausted,
            # all text from the start_pos until the end will be returned
            out.extend(data[i:])
            i = len(data)

    return out

def main():
    import sys
    data = [l.strip() for l in open(sys.argv[1]) if len(l.strip()) > 0]
    process_tranx_log(data)

if __name__ == '__main__':
    main()
