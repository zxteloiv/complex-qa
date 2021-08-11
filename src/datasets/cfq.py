from typing import Optional
from trialbot.data import JsonLDataset, PickleDataset, RedisDataset, IndexDataset
from .lark_dataset import LarkGrammarDataset, LarkParserDatasetWrapper
from trialbot.utils.root_finder import find_root
import json
from os.path import join

ROOT = find_root()
CFQ_PATH = join(ROOT, 'data', 'cfq')
GRAMMAR_PATH = join(ROOT, 'src', 'statics', 'grammar')
NEW_GRAMMAR = join('run', 'cfq_mcd1.30.lark')

REDIS_CONN_C = ('localhost', 6379, 0)
REDIS_CONN_S = ('localhost', 6379, 1)


def _get_split(filename: str) -> dict:
    all_idx = json.load(open(join(CFQ_PATH, 'splits', filename)))
    split_idx = {
        'train': all_idx['trainIdxs'],
        'dev': all_idx['devIdxs'],
        'test': all_idx['testIdxs'],
    }
    return split_idx

def cfq_preparsed_treebase(split_filename, conn=REDIS_CONN_C):
    import lark
    store = PickleDataset(join(CFQ_PATH, 'parsed_cfq.pkl'), conn, 'cfq_parse_')
    split_idx = _get_split(split_filename)
    return tuple(IndexDataset(store, split_idx[tag]) for tag in ['train', 'dev', 'test'])

def cfq_simplified_treebase(split_filename: str, parse_keys):
    grammar_file = NEW_GRAMMAR
    store = JsonLDataset(join(CFQ_PATH, 'dataset_slim.jsonl.gz'))
    split_idx = _get_split(split_filename)
    grammar_tag = grammar_file[grammar_file.rfind('/') + 1:grammar_file.index('.lark')].lower()
    ds_tag = split_filename[:split_filename.index('.json')].lower()

    def _get_ds(split_tag):
        return RedisDataset(
            dataset=LarkParserDatasetWrapper(
                grammar_filename=grammar_file,
                startpoint='queryunit',
                parse_keys=parse_keys,
                dataset=IndexDataset(store, split_idx[split_tag]),
            ),
            conn=REDIS_CONN_S,
            prefix=f'{split_tag}_{ds_tag}_{grammar_tag}_',
        )

    def _get_ds_shared(split_tag):
        return IndexDataset(
            RedisDataset(
                LarkParserDatasetWrapper(
                    grammar_filename=grammar_file,
                    startpoint='queryunit',
                    parse_keys=parse_keys,
                    # read trees from the original dataset to parse, no split here, the db is for full parse cache
                    dataset=PickleDataset(join(CFQ_PATH, 'parsed_cfq.pkl'), REDIS_CONN_C, 'cfq_parse_')
                ),
                conn=REDIS_CONN_S,  # another conn to parse with other grammars
                prefix=f'{grammar_tag}'
            ),
            split_idx[split_tag],
        )

    return tuple(map(_get_ds_shared, ['train', 'dev', 'test']))

def cfq_iid():
    return cfq_preparsed_treebase('random_split.json')

def cfq_debug():
    import lark
    store = PickleDataset(join(CFQ_PATH, 'parsed_cfq.pkl'), ('localhost', 6379, 0), 'cfq_parse_')
    split_idx = _get_split('mcd1.json')
    split_idx['test'] = [7901, 178847, 46501, 77066]
    return tuple(IndexDataset(store, split_idx[tag]) for tag in ['train', 'dev', 'test'])

def cfq_mcd1_classic():
    return cfq_preparsed_treebase('mcd1.json')

def cfq_mcd2_classic():
    return cfq_preparsed_treebase('mcd2.json')

def cfq_mcd3_classic():
    return cfq_preparsed_treebase('mcd3.json')

def cfq_mcd1_simplified():
    return cfq_simplified_treebase('mcd1.json', ['sparqlPatternModEntities'])

def cfq_mcd2_simplified():
    return cfq_simplified_treebase('mcd2.json', ['sparqlPatternModEntities'])

def cfq_mcd3_simplified():
    return cfq_simplified_treebase('mcd3.json', ['sparqlPatternModEntities'])

def sparql_grammar(filename='sparql.bnf.lark'):
    file = join(GRAMMAR_PATH, filename)
    print("Dataset Filename:", file)
    d = LarkGrammarDataset(file, 'queryunit')
    return d, d, d

def sparql_pattern_grammar():
    return sparql_grammar('sparql_pattern.bnf.lark')

def get_cfq_data() -> dict:
    ds = dict()
    funcs = [
        (cfq_iid,  'cfq_iid_classic'),
        (cfq_mcd1_classic, 'cfq_mcd1_classic'),
        (cfq_mcd2_classic, 'cfq_mcd2_classic'),
        (cfq_mcd3_classic, 'cfq_mcd3_classic'),
        (cfq_mcd1_simplified, 'cfq_mcd1_simplified'),
        (cfq_mcd2_simplified, 'cfq_mcd2_simplified'),
        (cfq_mcd3_simplified, 'cfq_mcd3_simplified'),
    ]
    for pair in funcs:
        ds[pair[1]] = pair[0]
    return ds

def install_cfq_to_trialbot():
    from trialbot.training import Registry
    for k, v in get_cfq_data().items():
        Registry._datasets[k] = v
