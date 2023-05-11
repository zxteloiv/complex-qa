from functools import partial

from trialbot.data import JsonLDataset
from trialbot.utils.root_finder import find_root
import os.path as osp

DATA_PATH = osp.join(find_root('.ROOT'), 'data', 'cofe')

CATEGORIES = [
    'full_similarity',
    'high_complexity',
    'high_diversity',
    'low_complexity',
    'only_primitive_coverage',
    'rough_structural_similariy',
]


def get_ds(cat: str):
    ds = JsonLDataset(osp.join(DATA_PATH, cat + '.json'))
    return ds, ds, ds


def install_datasets(reg: dict = None):
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._datasets

    for i, cat in enumerate(CATEGORIES):
        reg[f'cofe_cat{i}'] = partial(get_ds, cat=cat)
