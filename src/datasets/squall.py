from trialbot.data import JsonDataset

from trialbot.utils.root_finder import find_root
from os.path import join

ROOT = find_root()
SQUALL_PATH = join(ROOT, 'data', 'squall', 'data')


def get_data_split_func(i: int):
    def dataset_fun():
        train = JsonDataset(join(SQUALL_PATH, f'train-{i}.json'))
        dev = JsonDataset(join(SQUALL_PATH, f'dev-{i}.json'))
        test = JsonDataset(join(SQUALL_PATH, f'wtq-test.json'))
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
