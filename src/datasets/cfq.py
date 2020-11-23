from trialbot.training import Registry
from trialbot.data.datasets.jsonl_dataset import JsonLDataset
from .lark_dataset import LarkDataset
from .index_dataset import IndexDataset
from utils.root_finder import find_root
import json
from os.path import join


ROOT = find_root()
CFQ_PATH = join(ROOT, 'data', 'cfq')

def cfq_filebase(split_filename):
    store = JsonLDataset(join(CFQ_PATH, 'dataset_slim.jsonl'))
    all_idx = json.load(open(split_filename))
    split_idx = list(map(all_idx.get, ['trainIdxs', 'devIdxs', 'testIdxs']))
    return tuple(IndexDataset(store, ds) for ds in split_idx)

@Registry.dataset()
def cfq_mcd1():
    splitfile = join(CFQ_PATH, 'splits', 'mcd1.json')
    return cfq_filebase(splitfile)

@Registry.dataset()
def cfq_mcd2():
    splitfile = join(CFQ_PATH, 'splits', 'mcd2.json')
    return cfq_filebase(splitfile)

@Registry.dataset()
def cfq_mcd3():
    splitfile = join(CFQ_PATH, 'splits', 'mcd3.json')
    return cfq_filebase(splitfile)

@Registry.dataset('sparql')
def sparql_grammar(filename='sparql.bnf.lark'):
    file = join(ROOT, 'src', 'statics', 'grammar', 'sparql.bnf.lark')
    d = LarkDataset(file, 'queryunit')
    return d, d, d

@Registry.dataset('sparql_pattern')
def sparql_pattern_grammar():
    return sparql_grammar('sparql_pattern.bnf.lark')

