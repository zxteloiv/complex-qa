from trialbot.training import Registry
from .json_dataset import JsonDataset
from .index_dataset import IndexDataset
from utils.root_finder import find_root
import json
from os.path import join


ROOT = find_root()
CFQ_PATH = join(ROOT, 'data', 'cfq')

def cfq_filebase(split_filename):
    store = JsonDataset(join(CFQ_PATH, 'dataset_slim.json'))
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
