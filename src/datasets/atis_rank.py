from trialbot.training import Registry
from .jsonl_dataset import JsonLDataset
from utils.root_finder import find_root

import os.path
_DATA_PATH = os.path.join(find_root(), 'data')
_ATIS_DATA = os.path.join(_DATA_PATH, 'atis_rank')

def atis_pure_none():
    train_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.none_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.none_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.none_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('atis_none_hyp')
def atis_none():
    train_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.none_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.five_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.five_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('atis_five_hyp')
def atis_five():
    train_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.five_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.five_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.five_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('atis_ten_hyp')
def atis_ten():
    train_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.ten_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.ten_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.ten_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('atis_ten_hyp_five_test')
def atis_ten_five():
    train_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.ten_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.ten_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.five_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('atis_full_hyp')
def atis_full():
    train_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.full_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.full_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_ATIS_DATA, 'atis_rank.full_hyp.test.jsonl'))
    return train_data, dev_data, test_data


