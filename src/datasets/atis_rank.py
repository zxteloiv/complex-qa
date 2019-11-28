from trialbot.training import Registry
from .jsonl_dataset import JsonLDataset
from utils.root_finder import find_root

import os.path
_DATA_PATH = os.path.join(find_root(), 'data')

@Registry.dataset('atis_none_hyp')
def atis_none():
    train_data = JsonLDataset(os.path.join(_DATA_PATH, 'atis_rank', 'none_hyp', 'atis_rank.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DATA_PATH, 'atis_rank', 'none_hyp', 'atis_rank.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DATA_PATH, 'atis_rank', 'none_hyp', 'atis_rank.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('atis_five_hyp')
def atis_five():
    train_data = JsonLDataset(os.path.join(_DATA_PATH, 'atis_rank', 'top_five_hyp', 'atis_rank.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DATA_PATH, 'atis_rank', 'top_five_hyp', 'atis_rank.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DATA_PATH, 'atis_rank', 'top_five_hyp', 'atis_rank.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('atis_full_hyp')
def atis_full():
    train_data = JsonLDataset(os.path.join(_DATA_PATH, 'atis_rank', 'full_hyp', 'atis_rank.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DATA_PATH, 'atis_rank', 'full_hyp', 'atis_rank.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DATA_PATH, 'atis_rank', 'full_hyp', 'atis_rank.test.jsonl'))
    return train_data, dev_data, test_data





