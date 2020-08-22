from trialbot.training import Registry
from trialbot.data.datasets.jsonl_dataset import JsonLDataset
from utils.root_finder import find_root

import os.path
_DATA_PATH = os.path.join(find_root(), 'data')
_DJANGO = os.path.join(_DATA_PATH, 'django_rank')

def django_pure_none():
    train_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.none_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.none_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.none_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('django_none_hyp')
def django_none():
    train_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.none_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.fifteen_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.fifteen_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('django_fifteen_hyp')
def django_15():
    train_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.fifteen_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.fifteen_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.fifteen_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('django_thirty_hyp')
def django_30():
    train_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.thirty_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.thirty_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.thirty_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('django_thirty_hyp_fifteen_test')
def django_30_15():
    train_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.thirty_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.thirty_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.fifteen_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('django_five_hyp')
def django_five():
    train_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.five_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.five_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.five_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('django_full_hyp')
def django_full():
    train_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.full_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.full_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.full_hyp.test.jsonl'))
    return train_data, dev_data, test_data

@Registry.dataset('django_test_on_training')
def django_test_on_training():
    train_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.five_hyp.train.jsonl'))
    dev_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.five_hyp.dev.jsonl'))
    test_data = JsonLDataset(os.path.join(_DJANGO, 'django_rank.five_hyp.train.jsonl'))
    return train_data, dev_data, test_data

