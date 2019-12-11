# Construct similarity matrix by BERT and natural language
# For every dataset

import pickle
import logging
logging.basicConfig(level=logging.INFO)

from collections import defaultdict
import apted
from apted.helpers import Tree
import astor
import ast

class ConstructTreeSimilarity:
    def __init__(self, dataset):
        self.logger = logging.getLogger(__name__)
        if dataset == 'atis':
            self.dump_code = ConstructTreeSimilarity.dump_lambda
        elif dataset == 'django':
            self.dump_code = ConstructTreeSimilarity.dump_python_ast_tree
        else:
            pass
        self.dataset = dataset

    @staticmethod
    def dump_python_ast_tree(s):
        return s.replace('(', '{').replace(')', '}')

    @staticmethod
    def dump_python(code, special=ast.AST):
        node = ast.parse(code)
        def dump(node, name=None):
            name = name or ''
            values = list(astor.iter_node(node))
            if isinstance(node, list):
                prefix, suffix = '{%s' % name, '}'
            elif values:
                prefix, suffix = '{%s' % type(node).__name__, '}'
            elif isinstance(node, special):
                prefix, suffix = '{%s' % name + type(node).__name__, '}'
            else:
                return '{%s}' % type(node).__name__
            node = [dump(a, b) for a, b in values if b != 'ctx']
            return '%s %s %s' % (prefix, ' '.join(node), suffix)
        return dump(node)

    @staticmethod
    def dump_lambda(s):
        s = s.strip()
        if not s.startswith('('):
            s = '( ' + s + ' )'

        tokens = s.split(' ')
        for i, tok in enumerate(tokens[1:], start=1):
            if tokens[i - 1] != '(' and tok not in ('(', ')'):
                tokens[i] = '(%s)' % tok

        out = ' '.join(tokens).replace('(', '{').replace(')', '}')
        return out

    def get_sim_matrix(self, dataset, compared_with):
        sim_lookup = defaultdict(list)
        for example in dataset:
            eid, hyp_rank = example['ex_id'], example['hyp_rank']
            query_key = f"{eid}-{hyp_rank}"

            self.logger.info(f'current progress={query_key}')

            try:
                query_code = self.dump_code(example["hyp"])
            except:
                query_code = self.dump_code(example["tgt"])

            distances = defaultdict(list)
            for support in compared_with:
                # any hyp must be matched with tgt, since only similar hyp doesn't mean anything
                try:
                    tgt = self.dump_code(support["tgt"])
                    d = apted.APTED(Tree.from_text(query_code), Tree.from_text(tgt),
                                    config=apted.PerEditOperationConfig(1., 1., 1.)).compute_edit_distance()
                except:
                    d = 2147483647
                distances[support['ex_id']].append(d)
            nearest_ex = sorted(distances.keys(), key=lambda k: distances[k][0])[:30]
            sim_lookup[query_key] = nearest_ex

        return sim_lookup

    def compare_pairs(self, dataset, compared_with):
        for example in dataset:
            eid, hyp_rank = example['ex_id'], example['hyp_rank']
            query_key = f"{eid}-{hyp_rank}"

            self.logger.info(f'current progress={query_key}')

            def _get_tgt(e):
                return e['tgt'] if self.dataset == 'atis' else e['tgt_tree']

            query_code = self.dump_code(_get_tgt(example))
            for support in compared_with:
                tgt = self.dump_code(_get_tgt(support))
                yield {"testID": f"{query_key}-{support['ex_id']}", "t1": query_code, "t2": tgt, "d": 3}

    def _get_dataset(self):
        # get rid of trialbot dependency
        def get_json_data(json_file):
            import json
            return [json.loads(l) for l in open(json_file).readlines()]

        if self.dataset == 'atis':
            prefix='../../data/atis_rank'
            train_set = get_json_data(os.path.join(prefix, 'atis_rank.five_hyp.train.jsonl'))
            dev_set = get_json_data(os.path.join(prefix, 'atis_rank.five_hyp.dev.jsonl'))
            test_set = get_json_data(os.path.join(prefix, 'atis_rank.five_hyp.test.jsonl'))
            support_set = get_json_data(os.path.join(prefix, 'atis_rank.none_hyp.train.jsonl'))

        elif self.dataset == 'django':
            prefix='../../data/django_rank'
            train_set = get_json_data(os.path.join(prefix, 'django_rank.none_hyp.train.jsonl'))
            dev_set = get_json_data(os.path.join(prefix, 'django_rank.none_hyp.dev.jsonl'))
            test_set = get_json_data(os.path.join(prefix, 'django_rank.none_hyp.test.jsonl'))
            support_set = get_json_data(os.path.join(prefix, 'django_rank.none_hyp.train.jsonl'))

        else:
            raise ValueError
        return train_set, dev_set, test_set, support_set

    def build(self, query_type):

        train_set, dev_set, test_set, support_set = self._get_dataset()
        log = self.logger
        if query_type == 'train':
            log.info("Querying the index with the training dataset")
            mat = self.get_sim_matrix(train_set, support_set)

        elif query_type == "test":
            log.info("Querying the index with the testing dataset")
            mat = self.get_sim_matrix(test_set, support_set)

        elif query_type == "dev":
            log.info("Querying the index with the dev dataset")
            mat = self.get_sim_matrix(dev_set, support_set)

        return mat

    def dump_compare(self, query_type):
        train_set, dev_set, test_set, support_set = self._get_dataset()
        log = self.logger
        if query_type == 'train':
            log.info("Querying the index with the training dataset")
            yield from self.compare_pairs(train_set, support_set)

        elif query_type == "test":
            log.info("Querying the index with the testing dataset")
            yield from self.compare_pairs(test_set, support_set)

        elif query_type == "dev":
            log.info("Querying the index with the dev dataset")
            yield from self.compare_pairs(dev_set, support_set)

        return

if __name__ == '__main__':
    import sys
    sys.path = ['..'] + sys.path
    import os.path
    from utils.root_finder import find_root
    sim_dir = os.path.join(find_root(), 'data', '_similarity_index')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['atis', 'django'])
    parser.add_argument('--query-type', type=str, choices=['train', 'dev', 'test'])
    parser.add_argument('--output-dir', type=str, default=sim_dir)
    parser.add_argument('--dump-cases', action="store_true")
    args = parser.parse_args()

    cons = ConstructTreeSimilarity(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dump_cases:
        compare_json = cons.dump_compare(args.query_type)
        with open(f'{args.dataset}_ted_lf_{args.query_type}_pairs.json', 'w') as f:
            import json
            for obj in compare_json:
                print(json.dumps(obj), file=f)

    else:
        mats = cons.build(args.query_type)
        ted_sim_path = os.path.join(args.output_dir, f'{args.dataset}_ted_lf_{args.query_type}.bin')
        cons.logger.info('Dump the similarity lookup table...')
        pickle.dump(mats, open(ted_sim_path, 'wb'))



