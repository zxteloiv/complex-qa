import sys
sys.path = ['..'] + sys.path

from tqdm import tqdm
import pickle
import os.path
import logging
import multiprocessing
from collections import defaultdict
from utils.root_finder import find_root
logging.basicConfig(level=logging.INFO)
from utils.code_transformation import CodeTransform
from utils.ir_client import SolrClient
from utils.merge_sort_parallel import merge_sort_parallel
from functools import partial

def nl_ngram(args):
    if args.dataset == "atis":
        from datasets.atis_rank import atis_pure_none
        fn_load_data = atis_pure_none
        core = "atis_none_lf"
    elif args.dataset == "django":
        from datasets.django_rank import django_pure_none
        fn_load_data = django_pure_none
        core = "django_none_lf"
    else:
        raise ValueError("dataset not found")

    idx = SolrClient(core)
    datasets = fn_load_data()

    # find similarities for training, dev, and test set
    output = []
    for dataset in datasets:
        ds_rtn = []
        for ex in tqdm(dataset, total=len(dataset)):  # datasets are all ensured to be ordered and to start from 0
            text = idx.escape(' '.join(ex['src']))
            query = {"q": text, "wt": "json", "rows": 30,
                     "df": "hyp", 'fl': 'id, score', 'sort': 'score desc'}
            try:
                res = idx.search(query)['response']['docs']
            except:
                logging.warning(f"Error Request: ID={ex['ex_id']} Text={text}")
                continue
            similar_ids = [r['id'] for r in res]
            ds_rtn.append(similar_ids)
        output.append(ds_rtn)

    outfile = os.path.join(args.output_dir, "{0}_{1}.bin".format(args.dataset, args.action))
    pickle.dump(output, open(outfile, 'wb'))

def lf_ngram(args):
    if args.dataset == "atis":
        from datasets.atis_rank import atis_five
        fn_load_data = atis_five
        core = "atis_5_lf"
    elif args.dataset == "django":
        from datasets.django_rank import django_15
        fn_load_data = django_15
        core = "django_15_lf"
    else:
        raise ValueError("dataset not found")

    idx = SolrClient(core)
    datasets = fn_load_data()

    # find similarities for training, dev, and test set
    output = []
    from datasets.cached_retriever import get_hyp_key
    for dataset in datasets:
        ds_rtn = defaultdict(list)
        for ex in tqdm(dataset, total=len(dataset)):
            text = idx.escape(ex['hyp'])
            query = {"q": text, "wt": "json", "rows": 30,
                     "df": "hyp", 'fl': 'id, score', 'sort': 'score desc'}
            try:
                res = idx.search(query)['response']['docs']
            except:
                logging.warning(f"Error Request: ID={get_hyp_key(ex)} Text={text}")
                continue

            similar_keys = [r['id'] for r in res]
            ds_rtn[get_hyp_key(ex)].append(similar_keys)
        output.append(ds_rtn)

    outfile = os.path.join(args.output_dir, "{0}_{1}.bin".format(args.dataset, args.action))
    pickle.dump(output, open(outfile, 'wb'))

def lf_ted(args):
    if args.dataset == "atis":
        from datasets.atis_rank import atis_five as fn_load_data
        core, ted_key, transform = "atis_5_lf", "hyp", CodeTransform.dump_lambda
    elif args.dataset == "django":
        from datasets.django_rank import django_15 as fn_load_data
        core, ted_key, transform = "django_15_lf", "hyp_tree", CodeTransform.dump_python_ast_tree
    else:
        raise ValueError("dataset not found")

    idx = SolrClient(core)
    datasets = fn_load_data()

    # find similarities for training, dev, and test set
    output = []
    from datasets.cached_retriever import get_hyp_key
    for dataset in datasets:
        ds_rtn = defaultdict(list)
        pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 6))
        # dataset = dataset[:100]
        for ex in tqdm(dataset, total=len(dataset)):
            text = idx.escape(ex['hyp'])
            query = {"q": text, "wt": "json",
                     "rows": 50 if args.dataset == "atis" else 60,
                     "df": "hyp", 'fl': 'id, hyp, hyp_tree, score', 'sort': 'score desc'}
            try:
                candidates = idx.search(query)['response']['docs']
            except KeyboardInterrupt:
                return
            except:
                logging.warning(f"Error Request: ID={get_hyp_key(ex)} Text={text}")
                continue

            comp_key = partial(get_ted, ex=ex, transform=transform, ted_key=ted_key)
            reranking = sorted(candidates, key=comp_key)
            # reranking = merge_sort_parallel(candidates, key=comp_key, max_pool_size=6, pool=pool)
            similar_keys = [r['id'] for r in reranking][:30]
            ds_rtn[get_hyp_key(ex)].append(similar_keys)
        output.append(ds_rtn)

    outfile = os.path.join(args.output_dir, "{0}_{1}.bin".format(args.dataset, args.action))
    pickle.dump(output, open(outfile, 'wb'))

# to run Pool.map for the candidates, they must be defined as a function object rather than lambda exp.
def get_ted(c, ex, transform, ted_key):
    import apted
    from apted.helpers import Tree
    try:
        t1, t2 = c[ted_key][0], ex[ted_key]
        t1, t2 = list(map(transform, (t1, t2)))
        ted = apted.APTED(Tree.from_text(t1), Tree.from_text(t2),
                          config=apted.PerEditOperationConfig(1., 1., 1.))
        d = ted.compute_edit_distance()
    except KeyboardInterrupt:
        raise KeyboardInterrupt()
    except:
        d = 2147483647
    return d

def main():
    import os.path
    sim_dir = os.path.join(find_root(), 'data', '_similarity_index')

    supported_action = ['nl_ngram', 'lf_ngram', 'lf_ted']

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True, choices=supported_action)
    parser.add_argument('--dataset', type=str, required=True, choices=['atis', 'django'])
    parser.add_argument('--output-dir', type=str, default=sim_dir)
    args = parser.parse_args()

    func = eval(args.action)
    func(args)

if __name__ == '__main__':
    main()
