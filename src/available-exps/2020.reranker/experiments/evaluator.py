from typing import Optional
from trialbot.utils.file_reader import open_json
from collections import defaultdict
import json
import numpy as np

def _aggregate_data(test_file, ranking_file, max_rank_filter):
    test_data = list(open_json(test_file))
    rank_data = list(open_json(ranking_file))

    # data = defaultdict(list)
    # for example, score in zip(test_data, rank_data):
    #     assert all(example[k] == score[k] for k in ("ex_id", "hyp_rank"))
    #     example['ranking_score'] = score['ranking_score']
    #     data[example["ex_id"]].append(example)

    score_kv = defaultdict(lambda: {"ranking_score": -999999})
    for score in rank_data:
        k = f"{score['ex_id']}-{score['hyp_rank']}"
        score_kv[k] = score

    data = defaultdict(list)
    for example in test_data:
        if example['hyp_rank'] >= max_rank_filter:
            continue

        k = f"{example['ex_id']}-{example['hyp_rank']}"
        example['ranking_score'] = score_kv[k]['ranking_score']
        subrankings = list(filter(None, map(score_kv[k].get, ('rank_match', 'rank_a2b', 'rank_b2a'))))
        if len(subrankings) > 0:
            example['subrankings'] = subrankings
        data[example['ex_id']].append(example)
    return data

def _print_stat(data, rank_weights: Optional[list] = None):
    stat = defaultdict(lambda : 0)
    if rank_weights is None:
        keyfunc = lambda x: x['ranking_score']
    else:
        keyfunc = lambda x: np.dot(rank_weights, x['subrankings']).item() if 'subrankings' in x else -999999.

    if rank_weights is not None:
        print(f"using given weights: {rank_weights}")

    for i, xs in data.items():
        if len(xs) == 0:
            continue

        stat["example_count"] += 1
        stat["raw_top_1"] += 1 if xs[0]['is_correct'] else 0

        rerank = sorted(xs, key=keyfunc, reverse=True)
        stat["ranked_top_1"] += 1 if rerank[0]['is_correct'] else 0

    print("total = %d" % stat["example_count"])
    print("raw@1 = %d" % stat["raw_top_1"],
          ("(%.4f%%)" % (stat["raw_top_1"] / stat["example_count"]))
          if stat["example_count"] > 0 else "(n/a percentage)")
    print("rerank@1 = %d" % stat["ranked_top_1"],
          ("(%.4f%%)" % (stat["ranked_top_1"] / stat["example_count"]))
          if stat["example_count"] > 0 else "(n/a percentage)")

def evaluate(test_file, ranking_file, max_rank_filter, weights):
    data = _aggregate_data(test_file, ranking_file, max_rank_filter)
    _print_stat(data, weights)

def dump_rerank(test_file, ranking_file, max_rank_filter):
    import json
    data = _aggregate_data(test_file, ranking_file, max_rank_filter)
    for k in sorted(data.keys()):
        xs = data[k]
        if len(xs) == 0:
            continue
        xs = sorted(xs, key=lambda x: x['ranking_score'], reverse=True)
        for rerank, x in enumerate(xs):
            x['rerank'] = rerank
            print(json.dumps(x))

def inspect_error(test_file, ranking_file, max_rank_filter):
    import json
    data = _aggregate_data(test_file, ranking_file, max_rank_filter)
    for k in sorted(data.keys()):
        xs = data[k]
        if len(xs) == 0:
            continue

        xs = sorted(xs, key=lambda x: x['ranking_score'], reverse=True)
        if not any(x['is_correct'] for x in xs):
            continue

        if xs[0]['is_correct']:
            continue

        for rerank, x in enumerate(xs):
            x['rerank'] = rerank
            print(json.dumps(x))


def _normalize_subrankings(data, rkey='subrankings'):
    # shape: (hyp_num, #subrankings=3)
    subranking_vecs = np.array(list(filter(None, (score.get(rkey) for v in data.values() for score in v))))
    mean = subranking_vecs.mean(axis=0).tolist()
    std = subranking_vecs.std(axis=0).tolist()

    for k, v in data.items():           # key: ex_id
        for ith, item in enumerate(v):  # item: ex_id, hyp
            subranks = item[rkey] if rkey in item else [-99999, -99999, -99999]
            norm_ranks = list(map(lambda r, m, s: (r - m) / s, subranks, mean, std))
            data[k][ith][rkey]  = norm_ranks


def min_dev(test_file, ranking_file, max_rank_filter):
    """find hyperparameters on given ranking file"""
    import json
    import numpy as np
    from scipy.optimize import minimize
    data = _aggregate_data(test_file, ranking_file, max_rank_filter)
    _normalize_subrankings(data)

    coef_m, coef_a2b, coef_b2a = 0, 0, 0
    for k in sorted(data.keys()):
        xs = data[k]
        if len(xs) <= 1:
            continue

        xs = sorted(xs, key=lambda x: x['is_correct'], reverse=True)
        gold, others = xs[0], xs[1:]
        if gold['is_correct'] == 0:
            continue    # no gold answer found, the example is ignored, which will not affect hparam-search.

        gold_m, gold_a2b, gold_b2a = gold['subrankings']
        for other in others:
            other_m, other_a2b, other_b2a = other['subrankings']
            coef_m += other_m - gold_m
            coef_a2b += other_a2b - gold_a2b
            coef_b2a += other_b2a - gold_b2a

    def tgt_func(x, *args):
        return np.dot(x, args[:2]) + (1 - np.sum(x)) * args[2]

    bounds = [(0., 1.), (0., 1.)]
    init = np.array([.3333333, .3333333])
    print(f"minimize with coefs={[coef_m, coef_a2b, coef_b2a]}, bounds={bounds}")
    res = minimize(tgt_func, init, args=(coef_m, coef_a2b, coef_b2a), bounds=bounds)
    hp = res.x.tolist()
    print(f"best hyperparams:\n"
          f"rank_match = {hp[0]}\n"
          f"rank_a2b   = {hp[1]}\n"
          f"rank_b2a   = {1 - hp[0] - hp[1]}\n"
          )

    _print_stat(data, hp + [1 - hp[0] - hp[1]])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs=2, metavar=('TEST_FILE', 'SCORE_FILE'))
    parser.add_argument('--max-hyp-rank', default=30, type=int)
    parser.add_argument('--action', '-a', choices=["min_dev", "dump_rerank", "evaluate", "inspect_error"], default="evaluate")
    parser.add_argument('--weights', '-w', nargs=3, type=float)

    args = parser.parse_args()
    if args.action == 'evaluate':
        evaluate(*args.files, args.max_hyp_rank, args.weights)
    else:
        func = eval(args.action)
        func(*args.files, args.max_hyp_rank)
