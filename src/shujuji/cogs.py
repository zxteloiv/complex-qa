from functools import partial

from trialbot.data import TabSepFileDataset, RedisDataset
from trialbot.utils.root_finder import find_root

from shujuji.lark_dataset import LarkParserDatasetWrapper
from utils.trialbot.transform_dataset import TransformData
import os.path as osp

DATA_PATH = osp.join(find_root(), 'data', 'cogs', 'data')


def cogs_ds_from_file(filename):
    return TransformData(
        dataset=TabSepFileDataset(filename),
        transform_fn=lambda x: dict(zip(("nl", "lf", "tag"), x))
    )


COGS_GRAMMAR = osp.join(find_root('.SRC'), 'statics', 'grammar', 'cogs.lark')
REDIS_CONN = ('localhost', 36379, 0)


def parsed_cogs_from_file(filename,
                          parse_grammar: str = COGS_GRAMMAR,
                          start_nonterminal: str = 'start',
                          redis_conn: tuple = REDIS_CONN,
                          redis_prefix: str = 'cogs_',
                          redis_expire_sec: int = 0,
                          ):
    return RedisDataset(
        dataset=LarkParserDatasetWrapper(
            grammar_filename=parse_grammar,
            startpoint=start_nonterminal,
            parse_keys='lf',
            dataset=TransformData(
                dataset=TabSepFileDataset(filename),
                transform_fn=lambda x: dict(zip(("nl", "lf", "tag"), x)),
            ),
        ),
        conn=redis_conn,
        prefix=redis_prefix, # the prefix can be safely ignored without redis server
        expire_sec=redis_expire_sec,
    )


def cogs_iid():
    files = map(lambda n: osp.join(DATA_PATH, n), ('train.tsv', 'dev.tsv', 'test.tsv'))
    ds = list(map(cogs_ds_from_file, files))
    return ds


def cogs_gen():
    files = map(lambda n: osp.join(DATA_PATH, n), ('train.tsv', 'dev.tsv', 'gen.tsv'))
    ds = list(map(cogs_ds_from_file, files))
    return ds


def cogs_iid_parsed():
    files = map(lambda n: osp.join(DATA_PATH, n), ('train.tsv', 'dev.tsv', 'test.tsv'))
    train, dev, test = files
    ds = (
        parsed_cogs_from_file(train, redis_prefix='cogs_train_'),
        parsed_cogs_from_file(dev, redis_prefix='cogs_dev_'),
        parsed_cogs_from_file(test, redis_prefix='cogs_test_'),
    )
    return ds


def cogs_gen_parsed():
    files = map(lambda n: osp.join(DATA_PATH, n), ('train.tsv', 'dev.tsv', 'gen.tsv'))
    train, dev, test = files
    ds = (
        # In COGS, IID and GEN splits share the same training and dev data, and thus the redis tags
        parsed_cogs_from_file(train, redis_prefix='cogs_train_'),
        parsed_cogs_from_file(dev, redis_prefix='cogs_dev_'),
        parsed_cogs_from_file(test, redis_prefix='cogs_gen_'),
    )
    return ds


def install_dataset(reg_dict: dict = None):
    if reg_dict is None:
        from trialbot.training import Registry
        reg_dict = Registry._datasets

    reg_dict['cogs_iid'] = cogs_iid
    reg_dict['cogs_gen'] = cogs_gen
    reg_dict['cogs_iid_parsed'] = cogs_iid_parsed
    reg_dict['cogs_gen_parsed'] = cogs_gen_parsed
