from trialbot.data import JsonDataset

from trialbot.utils.root_finder import find_root
from os.path import join
import json

ROOT = find_root()
SQUALL_PATH = join(ROOT, 'data', 'squall', 'data')
SQUALL_TABLE = join(ROOT, 'data', 'squall', 'tables', 'json')


def _join_table(example):
    table_id = example.get('tbl')
    table_file = join(SQUALL_TABLE, f'{table_id}.json')
    example['tbl_cells'] = json.loads(open(table_file).read())
    return example


def get_data_split_func(i: int):
    def dataset_fun():
        from utils.trialbot.transform_dataset import TransformData
        train = TransformData(JsonDataset(join(SQUALL_PATH, f'train-{i}.json')), _join_table)
        dev = TransformData(JsonDataset(join(SQUALL_PATH, f'dev-{i}.json')), _join_table)
        test = TransformData(JsonDataset(join(SQUALL_PATH, f'wtq-test.json')), _join_table)
        return train, dev, test

    return dataset_fun


def install_squall_datasets(reg_dict=None):
    split_range = range(5)
    if reg_dict is None:
        from trialbot.training import Registry
        reg_dict = Registry._datasets

    for i in split_range:
        ds_name = f'squall{i}'
        reg_dict[ds_name] = get_data_split_func(i)
