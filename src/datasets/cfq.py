from typing import Optional
from trialbot.training import Registry
from trialbot.data.datasets.jsonl_dataset import JsonLDataset
from .lark_dataset import LarkGrammarDataset, LarkParserDatasetWrapper
from .pickle_dataset import PickleDataset
from trialbot.data.datasets.index_dataset import IndexDataset
from utils.root_finder import find_root
import json
from os.path import join

ROOT = find_root()
CFQ_PATH = join(ROOT, 'data', 'cfq')
GRAMMAR_PATH = join(ROOT, 'src', 'statics', 'grammar')

def cfq_filebase(split_filename):
    store = JsonLDataset(join(CFQ_PATH, 'dataset_slim.jsonl.gz'))
    all_idx = json.load(open(split_filename))
    split_idx = list(map(all_idx.get, ['trainIdxs', 'devIdxs', 'testIdxs']))
    return tuple(IndexDataset(store, ds) for ds in split_idx)

def cfq_treebase(split_filename, grammar_file: str, keys):
    store = JsonLDataset(join(CFQ_PATH, 'dataset_slim.jsonl.gz'))
    all_idx = json.load(open(split_filename))
    split_idx = list(map(all_idx.get, ['trainIdxs', 'devIdxs', 'testIdxs']))
    get_parse_tree_dataset = lambda d: LarkParserDatasetWrapper(grammar_file, 'queryunit', keys, d)
    return tuple(get_parse_tree_dataset(IndexDataset(store, ds)) for ds in split_idx)

def cfq_preparsed_treebase(split_filename):
    import lark
    store = PickleDataset(join(CFQ_PATH, 'parsed_cfq.pkl'), ('localhost', 6379, 0), 'cfq_parse_')
    all_idx = json.load(open(split_filename))
    split_idx = map(all_idx.get, ['trainIdxs', 'devIdxs', 'testIdxs'])
    return tuple(IndexDataset(store, ds) for ds in split_idx)

@Registry.dataset()
def cfq_iid():
    splitfile = join(CFQ_PATH, 'splits', 'random_split.json')
    return cfq_preparsed_treebase(splitfile)

@Registry.dataset()
def cfq_mcd1():
    splitfile = join(CFQ_PATH, 'splits', 'mcd1.json')
    return cfq_preparsed_treebase(splitfile)

@Registry.dataset()
def cfq_mcd2():
    splitfile = join(CFQ_PATH, 'splits', 'mcd2.json')
    return cfq_preparsed_treebase(splitfile)

@Registry.dataset()
def cfq_mcd3():
    splitfile = join(CFQ_PATH, 'splits', 'mcd3.json')
    return cfq_preparsed_treebase(splitfile)

@Registry.dataset()
def cfq_mcd1_test_on_training_set():
    import lark
    store = PickleDataset(join(CFQ_PATH, 'parsed_cfq.pkl'), ('localhost', 6379, 0), 'cfq_parse_')
    splitfile = join(CFQ_PATH, 'splits', 'mcd1.json')
    all_idx = json.load(open(splitfile))
    split_idx = map(all_idx.get, ['trainIdxs', 'devIdxs', 'trainIdxs'])
    return tuple(IndexDataset(store, ds) for ds in split_idx)

@Registry.dataset()
def cfq_mcd1_runtime_tree():
    splitfile = join(CFQ_PATH, 'splits', 'mcd1.json')
    grammar = join(GRAMMAR_PATH, 'sparql_pattern.bnf.lark')
    return cfq_treebase(splitfile, grammar, ['sparql', 'sparqlPattern', 'sparqlPatternModEntities'])

@Registry.dataset('sparql')
def sparql_grammar(filename='sparql.bnf.lark'):
    file = join(GRAMMAR_PATH, filename)
    print("Dataset Filename:", file)
    d = LarkGrammarDataset(file, 'queryunit')
    return d, d, d

@Registry.dataset('sparql_pattern')
def sparql_pattern_grammar():
    return sparql_grammar('sparql_pattern.bnf.lark')

