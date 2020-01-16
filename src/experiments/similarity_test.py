import sys
sys.path = ['..'] + sys.path
import os.path

import datasets.cached_retriever as libretriever
from datasets.atis_rank import atis_five

from utils.root_finder import find_root

ROOT = find_root()

sim_dir = os.path.join(find_root(), 'data', '_similarity_index')

train_set, dev_set, test_set = atis_five()

nl_ret = libretriever.IDCacheRetriever(os.path.join(sim_dir, "atis_nl_ngram.bin"), train_set)
lf_ret = libretriever.HypIDCacheRetriever(
    os.path.join(sim_dir, "atis_lf_ngram.bin"),
    train_set
)
ted_ret = libretriever.HypIDCacheRetriever(
    os.path.join(sim_dir, "atis_lf_ted.bin"),
    train_set
)

example = dev_set[17]
print(example)
print('=' * 64)
for ret in (nl_ret, lf_ret, ted_ret):
    all_res = ret.search(example, "dev")
    for res in all_res:
        print(f" id: {res['ex_id']}-{res['hyp_rank']},\n"
              f"src: {' '.join(res['src'])},\n"
              f"tgt: {res['tgt']},\n"
              f"hyp: {res['hyp']},\n"
              )
    print('-' * 64)
