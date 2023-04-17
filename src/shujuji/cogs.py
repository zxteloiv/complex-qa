from trialbot.data import TabSepFileDataset
from trialbot.utils.root_finder import find_root
from utils.trialbot.transform_dataset import TransformData
import os.path as osp

DATA_PATH = osp.join(find_root(), 'data', 'cogs', 'data')


def cogs_ds_from_file(filename):
    return TransformData(
        dataset=TabSepFileDataset(filename),
        transform_fn=lambda x: dict(zip(("nl", "lf", "tag"), x))
    )


def cogs_iid():
    files = map(lambda n: osp.join(DATA_PATH, n), ('train.tsv', 'dev.tsv', 'test.tsv'))
    ds = list(map(cogs_ds_from_file, files))
    return ds


def cogs_gen():
    files = map(lambda n: osp.join(DATA_PATH, n), ('train.tsv', 'dev.tsv', 'gen.tsv'))
    ds = list(map(cogs_ds_from_file, files))
    return ds


def install_dataset(reg_dict: dict = None):
    if reg_dict is None:
        from trialbot.training import Registry
        reg_dict = Registry._datasets

    reg_dict['cogs_iid'] = cogs_iid
    reg_dict['cogs_gen'] = cogs_gen
