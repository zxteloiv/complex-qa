import sys
from os.path import join
sys.path.insert(0, join('..', '..'))
import logging
logging.basicConfig(level=logging.INFO)
from utils.lark.stat import RuleCollector
from idioms.eval import step_evaluation
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_raw_sql_datasets()
cg_bundle.install_raw_qa_datasets()
import lark
from tqdm import tqdm
import io

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True,
                        help=f"valid datasets examples: {list(cg_bundle.CG_DATA_REG.keys())[:2]}"
                             f"... {list(cg_bundle.CG_DATA_REG.keys())[-2:]}")
    parser.add_argument('--grammar', '-g', help="something ends with .lark", required=True)
    parser.add_argument('--start', '-s', help="grammar start nonterminal", required=True)
    parser.add_argument('--key', '-k', default='sql', help='the key name in the dataset')
    parser.add_argument('--force-parsing', '-f', default=False, type=bool, help='force parsing the data, default 0')
    args = parser.parse_args()
    run_eval(args.dataset, args.grammar, args.start, args.key)

def run_eval(ds, grammar, start, key):
    train, dev, test = cg_bundle.CG_DATA_REG[ds]()
    c = RuleCollector()
    parser = lark.Lark(open(grammar), keep_all_tokens=True, start=start)
    train_trees = _parse(train, parser, key)
    dev_trees = _parse(dev, parser, key)
    test_trees = _parse(test, parser, key)
    print(f"train:\t{len(train_trees)}\t{len(train)}")
    print(f"dev:\t{len(dev_trees)}\t{len(dev)}")
    print(f"test:\t{len(test_trees)}\t{len(test)}")

    train_stat = c.run_for_statistics(0, train_trees)
    dev_stat = c.run_for_statistics(0, dev_trees)
    eval_results = step_evaluation(train_stat, dev_stat, c.rule_to_id)

    with io.StringIO() as f:
        h1 = eval_results.keys()
        h1_size = [len(v.keys()) for v in eval_results.values()]
        for t, l in zip(h1, h1_size):
            f.write(t)
            while l > 0:
                l -= 1
                f.write('\t')
        f.write('\n')
        h2 = sum([list(k + '_' + h for h in v.keys()) for k, v in eval_results.items()], start=[])
        print("\t".join(h2), file=f)
        val_list = sum([list(v.values()) for v in eval_results.values()], start=[])
        print(('%.4f\t' * len(val_list)).rstrip() % tuple(val_list), file=f)
        print(f.getvalue())

def _parse(ds, parser, key):
    train_trees = []
    for x in tqdm(ds):
        try:
            tree = parser.parse(x[key])
        except:
            continue
        train_trees.append(tree)
    return train_trees


if __name__ == '__main__':
    main()