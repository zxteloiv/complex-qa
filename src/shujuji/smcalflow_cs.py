from functools import partial
from trialbot.utils.root_finder import find_root
from trialbot.data import JsonLDataset, RedisDataset
import os.path as osp

from shujuji.lark_dataset import LarkParserDatasetWrapper

DATA_PATH = osp.join(find_root('.ROOT'), 'data', 'SMCalFlow-CS')


def install(reg: dict = None):
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._datasets

    for n in COMP_EX_NUMS:
        reg[f'smc{n}'] = partial(smc_by_num, num=n)
        reg[f'smc{n}_parsed'] = partial(parsed_smc_by_num, num=n)
    reg[f'smc128iid'] = smc128_iid
    reg[f'smc128iid_parsed'] = parsed_smc128_iid


# the number of compositional examples added into the training set,
# thereby yielding a few-shot generalization setting.
# the setting of 0 example is hard and that of 8 examples is not explored in the original paper.
# refer to 10.18653/v1/2021.naacl-main.225 for more information.
COMP_EX_NUMS = [16, 32, 64, 128]


def smc_by_num(num: int):
    path = osp.join(DATA_PATH, f'source_domain_with_target_num{num}')
    return smc_by_path(path)


def smc128_iid():
    path = osp.join(DATA_PATH, f'source_domain_with_target_num128_iid_shuffled')
    return smc_by_path(path)


def smc_by_path(path: str):
    files = ('train.jsonl', 'valid.jsonl', 'test.jsonl')
    names = [osp.join(path, f) for f in files]
    return list(map(JsonLDataset, names))


GRAMMAR_FILE = osp.join(find_root('.SRC'), 'statics', 'grammar', 'lispress.lark')
REDIS_CONN = ('localhost', 36379, 0)


def parsed_smc_by_num(num: int,
                      parse_grammar: str = GRAMMAR_FILE,
                      start_nonterminal: str = 'start',
                      redis_conn: tuple = REDIS_CONN,
                      redis_expire_sec: int = 0,
                      ):
    return parsed_smc_by_path(
        path=osp.join(DATA_PATH, f'source_domain_with_target_num{num}'),
        redis_pref=f'smcal{num}_',
        parse_grammar=parse_grammar,
        start_nonterminal=start_nonterminal,
        redis_conn=redis_conn,
        redis_expire_sec=redis_expire_sec,
    )


def parsed_smc128_iid(parse_grammar: str = GRAMMAR_FILE,
                      start_nonterminal: str = 'start',
                      redis_conn: tuple = REDIS_CONN,
                      redis_expire_sec: int = 0,
                      ):
    path = osp.join(DATA_PATH, f'source_domain_with_target_num128_iid_shuffled')
    return parsed_smc_by_path(
        path=path,
        redis_pref=f'smcal128iid_',
        parse_grammar=parse_grammar,
        start_nonterminal=start_nonterminal,
        redis_conn=redis_conn,
        redis_expire_sec=redis_expire_sec,
    )


def parsed_smc_by_path(path: str,
                       redis_pref: str,
                       parse_grammar: str = GRAMMAR_FILE,
                       start_nonterminal: str = 'start',
                       redis_conn: tuple = REDIS_CONN,
                       redis_expire_sec: int = 0,
                       ):
    def _get_ds(file, pref):
        return RedisDataset(
            dataset=LarkParserDatasetWrapper(
                grammar_filename=parse_grammar,
                startpoint=start_nonterminal,
                parse_keys='plan',
                dataset=JsonLDataset(file),
            ),
            conn=redis_conn,
            prefix=pref,
            expire_sec=redis_expire_sec,
        )

    files = ('train.jsonl', 'valid.jsonl', 'test.jsonl')
    train, dev, test = [osp.join(path, f) for f in files]
    return (
        _get_ds(train, f'{redis_pref}train_'),
        _get_ds(dev, f'{redis_pref}dev_'),
        _get_ds(test, f'{redis_pref}test_'),
    )
