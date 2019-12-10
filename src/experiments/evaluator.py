
from trialbot.utils.file_reader import open_json
from collections import defaultdict

def _aggregate_data(test_file, ranking_file, max_rank_filter):
    test_data = list(open_json(test_file))
    rank_data = list(open_json(ranking_file))

    # data = defaultdict(list)
    # for example, score in zip(test_data, rank_data):
    #     assert all(example[k] == score[k] for k in ("ex_id", "hyp_rank"))
    #     example['ranking_score'] = score['ranking_score']
    #     data[example["ex_id"]].append(example)

    score_kv = defaultdict(lambda: -999999)
    for score in rank_data:
        k = f"{score['ex_id']}-{score['hyp_rank']}"
        v = score['ranking_score']
        score_kv[k] = v

    data = defaultdict(list)
    for example in test_data:
        if example['hyp_rank'] >= max_rank_filter:
            continue

        k = f"{example['ex_id']}-{example['hyp_rank']}"
        example['ranking_score'] = score_kv[k]
        data[example['ex_id']].append(example)
    return data

def _print_stat(data):
    stat = defaultdict(lambda : 0)
    for i, xs in data.items():
        if len(xs) == 0:
            continue

        stat["example_count"] += 1
        stat["raw_top_1"] += 1 if xs[0]['is_correct'] else 0

        rerank = sorted(xs, key=lambda x: x["ranking_score"], reverse=True)
        stat["ranked_top_1"] += 1 if rerank[0]['is_correct'] else 0

    print("total = %d" % stat["example_count"])
    print("raw@1 = %d" % stat["raw_top_1"],
          ("(%.4f%%)" % (stat["raw_top_1"] / stat["example_count"]))
          if stat["example_count"] > 0 else "(n/a percentage)")
    print("rerank@1 = %d" % stat["ranked_top_1"],
          ("(%.4f%%)" % (stat["ranked_top_1"] / stat["example_count"]))
          if stat["example_count"] > 0 else "(n/a percentage)")

def evaluate(test_file, ranking_file, max_rank_filter):
    data = _aggregate_data(test_file, ranking_file, max_rank_filter)
    _print_stat(data)

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

if __name__ == '__main__':
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs=2, metavar=['TEST_FILE', 'SCORE_FILE'])
    parser.add_argument('--max-hyp-rank', default=30, type=int)
    parser.add_argument('--dump-rank', action="store_true")
    args = parser.parse_args()
    if args.dump_rank:
        dump_rerank(*args.files, args.max_hyp_rank)

    else:
        evaluate(*args.files, args.max_hyp_rank)