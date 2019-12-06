
from trialbot.utils.file_reader import open_json
from collections import defaultdict

def evaluate(test_file, ranking_file):
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
        k = f"{example['ex_id']}-{example['hyp_rank']}"
        example['ranking_score'] = score_kv[k]
        data[example['ex_id']].append(example)

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

if __name__ == '__main__':
    import sys
    evaluate(sys.argv[1], sys.argv[2])