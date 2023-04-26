from functools import partial
from typing import Literal

from trialbot.utils.root_finder import find_root
from trialbot.data import JsonLDataset, RedisDataset
import os.path as osp

from shujuji.lark_dataset import LarkParserDatasetWrapper

DATA_PATH = osp.join(find_root('.ROOT'), 'data', 'compact_cfq')

MCDs = [1, 2, 3]

GRAMMAR_FILE = osp.join(find_root('.SRC'), 'statics', 'grammar', 'compact_cfq.lark')
REDIS_CONN = ('localhost', 36379, 0)


def get_ccfq(mcd: int):
    files = [osp.join(DATA_PATH, f'mcd{mcd}', x, f'{x}.jsonl') for x in ('train', 'dev', 'test')]
    ds = list(map(JsonLDataset, files))
    return ds


def get_parsed_ccfq(mcd: int,
                    parse_grammar: str = GRAMMAR_FILE,
                    start_nonterminal: str = 'start',
                    redis_conn: tuple = REDIS_CONN,
                    redis_expire_sec: int = 0,
                    ):
    basename = osp.basename(parse_grammar)
    gtag = basename[:basename.rfind('.lark')]
    pref = f'ccfq_mcd{mcd}_' if gtag == 'compact_cfq' else f'ccfq_mcd{mcd}_{gtag}_'

    def _get_ds(split_file: Literal["train", "dev", "test"]):
        file = osp.join(DATA_PATH, f'mcd{mcd}', split_file, f'{split_file}.jsonl')
        return RedisDataset(
            dataset=LarkParserDatasetWrapper(
                grammar_filename=parse_grammar,
                startpoint=start_nonterminal,
                parse_keys='target',
                dataset=JsonLDataset(file),
            ),
            conn=redis_conn,
            prefix=pref,
            expire_sec=redis_expire_sec,
        )

    return _get_ds('train'), _get_ds('dev'), _get_ds('test')


def install_dataset(reg: dict = None):
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._datasets

    for n in MCDs:
        reg[f'ccfq_mcd{n}'] = partial(get_ccfq, mcd=n)
        reg[f'ccfq_mcd{n}_parsed'] = partial(get_parsed_ccfq, mcd=n)
