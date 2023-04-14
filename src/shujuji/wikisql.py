from functools import partial
from typing import Dict, Any

from trialbot.data.datasets.jsonl_dataset import JsonLDataset
from trialbot.utils.root_finder import find_root
import os.path as osp

from utils.trialbot.transform_dataset import TransformData


def get_wikisql():
    datapath = osp.join(find_root(), 'data', 'WikiSQL', 'data')

    def _get_tables(table_name) -> Dict[str, Dict[str, Any]]:
        return dict((ex['id'], ex) for ex in JsonLDataset(table_name))

    def _table_join(example, tables):
        example['table'] = tables[example['table_id']]
        return example

    def _get_ds(datafile, tablefile):
        tables = _get_tables(tablefile)
        ds = TransformData(JsonLDataset(datafile), transform_fn=partial(_table_join, tables=tables))
        return ds

    train_ds = _get_ds(osp.join(datapath, 'train.jsonl'), osp.join(datapath, 'train.tables.jsonl'))
    dev_ds = _get_ds(osp.join(datapath, 'dev.jsonl'), osp.join(datapath, 'dev.tables.jsonl'))
    test_ds = _get_ds(osp.join(datapath, 'test.jsonl'), osp.join(datapath, 'test.tables.jsonl'))
    return train_ds, dev_ds, test_ds


def install_dataset(reg_dict: dict = None):
    if reg_dict is None:
        from trialbot.training import Registry
        reg_dict = Registry._datasets

    reg_dict['wikisql'] = get_wikisql
