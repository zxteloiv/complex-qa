# ##
# The bundle of classic datasets re-split for compositional generalization evaluation.
# introduced by the paper Oren et al. 2020.
# Improving Compositional Generalization in Semantic Parsing
# https://www.aclweb.org/anthology/2020.findings-emnlp.225/
# https://github.com/inbaroren/improving-compgen-in-semparse
# ##
from itertools import product

from trialbot.data import JsonDataset
from trialbot.data import CompositionalDataset, RedisDataset

from utils.trialbot.chain_dataset import ChainDataset
from utils.trialbot.volatile_mem import VolatileDataset
from utils.trialbot.id_supplement_dataset import IDSupplement
from utils.trialbot.transform_dataset import TransformData
from utils.lark.id_tree import build_from_lark_tree
from .lark_dataset import LarkParserDatasetWrapper
from trialbot.utils.root_finder import find_root
import os
import os.path as osp
from os.path import join
from functools import partial
import logging
import re

ROOT = find_root()
SRC_PATH = find_root('.SRC')
CG_DATA_PATH = join(ROOT, 'data', 'CompGen', 'sql data')
CG_DATA_REG = dict()
REDIS_CONN: tuple = ('localhost', 36379, 2)


class FlattenSeqDS(CompositionalDataset):
    """
    Dataset reader for the specialized json structure in [Oren et al. 2020].
    In every example,
    each sentence (out of 1 or many) is paired with the only sql.
    These two sequences are anonymized into templates by substituting literal mentions into typed and ordered variables.
    The anonymization will definitely improve the results.

    Besides, the alignments and syntactic parse information are discarded.
    We use barely the two sequence.
    """
    def __init__(self, ds: JsonDataset, sql_only: bool = False):
        super().__init__(ds)
        self._flat_example_index = None   # list of pairs, each pair is (ds index, sent index, sql index).
        self.sql_only = sql_only    # used for SQL modeling, rather than conditional generation

    def _build_flat_index(self):
        if self._flat_example_index is not None:
            return

        self._flat_example_index = []
        for i in range(len(self.dataset)):
            if self.sql_only:
                for k in range(len(self.dataset[i]['sql'])):
                    self._flat_example_index.append((i, k))
            else:
                for j in range(sum([1 for sent in self.dataset[i]['sentences'] if sent['question-split'] != 'exclude'])):
                    # only the first sql will be used, so we don't have to save the index of sql instances.
                    self._flat_example_index.append((i, j))

    def __len__(self):
        self._build_flat_index()
        return len(self._flat_example_index)

    def get_example(self, i: int):
        self._build_flat_index()
        idx = self._flat_example_index[i]   # (i, j) or (i, k)
        group = self.dataset[idx[0]]
        if not self.sql_only:
            instance = {
                "sql": group['sql'][0],
                'ext_sql': group['sql'][1:],
                'sent': group['sentences'][idx[1]]['text'],
                'sql_vars': group['variables'],
                'sent_vars': group['sentences'][idx[1]]['variables'],
                'group_id': idx[0],
            }
        else:
            instance = {
                "sql": group['sql'][idx[1]],
                'variables': group['variables'],
                'group_id': idx[0],
            }

        sql = instance['sql']

        # adopt some of the preprocessing codes from Oren et al. 2020
        # only the mysql grammar accepts double quotes.
        sql = sql.replace("%", "").replace('"', "'").replace(",", " , ")

        # remove the table name with a fixed placeholder as in Oren et al. 2020,
        # the table alias actually starts with a table name, so its not required to generate.
        # the trick will significantly remove about 20 rules for terminals
        sql = re.sub(r" [A-Z_]+ (AS|as) ([A-Z_]+alias[0-9]) ", " TABLE_PLACEHOLDER AS \g<2> ", sql)

        instance['sql'] = sql
        return instance


def _get_grammar_tag_by_filename(grammar_file: str):
    grammar_tag = grammar_file[grammar_file.rfind('/') + 1:grammar_file.index('.lark')]
    grammar_tag = grammar_tag[grammar_tag.rfind('_') + 1:].lower()
    return grammar_tag


def _get_grammar_start(grammar_file: str):
    if 'sqlite' in grammar_file.lower():
        startpoint = 'parse'
    elif 'mysql' in grammar_file.lower():
        startpoint = 'query'
    elif 'handcrafted' in grammar_file.lower():
        startpoint = 'statement'
    else:
        raise ValueError(f"Unknown grammar file {grammar_file}")
    return startpoint


SPLIT_PATH = {
    'iid': 'new_question_split',
    'cg': 'schema_full_split'
}


DATA_PATH = {
    'ati': 'atis',
    'geo': 'geography',
    'adv': 'advising',
    'sch': 'scholar',
}


def _get_ds_full_dir(data_tag: str, split_tag: str):
    return join(CG_DATA_PATH, DATA_PATH[data_tag], SPLIT_PATH[split_tag])


def _get_raw_ds(ds_full_dir, *, sql_only: bool):
    def _build_ds(filename: str):
        ds = IDSupplement(FlattenSeqDS(JsonDataset(join(ds_full_dir, filename)), sql_only=sql_only))
        return ds

    train = _build_ds('aligned_train.json')
    dev = _build_ds('aligned_final_dev.json')
    test = _build_ds('final_test.json')
    logging.info(f"load dataset: {ds_full_dir}")
    return train, dev, test


def install_raw_sql_datasets(reg: dict = None):
    reg = CG_DATA_REG if reg is None else reg
    for ds_tag, split_tag in product(DATA_PATH.keys(), SPLIT_PATH.keys()):
        key = f"sql_{ds_tag}_{split_tag}"
        reg[key] = partial(_get_raw_ds, _get_ds_full_dir(ds_tag, split_tag), sql_only=True)
        logging.debug(f"registered {key} lazily")


def install_raw_qa_datasets(reg: dict = None):
    reg = CG_DATA_REG if reg is None else reg
    for ds_tag, split_tag in product(DATA_PATH.keys(), SPLIT_PATH.keys()):
        key = f"{ds_tag}_{split_tag}"
        reg[key] = partial(_get_raw_ds, _get_ds_full_dir(ds_tag, split_tag), sql_only=False)
        logging.debug(f"registered {key} lazily")


GRAMMAR_FILES = [
    join(SRC_PATH, 'statics', 'grammar', 'MySQL.lark'),
    join(SRC_PATH, 'statics', 'grammar', 'SQLite.lark'),
    join(SRC_PATH, 'statics', 'grammar', 'sql_handcrafted.lark'),
]


def _get_parsed_ds(ds_tag: str,
                   split_tag: str,
                   grammar_file: str,
                   sql_only: bool,
                   conn: tuple = REDIS_CONN,
                   ):
    def _add_runtime(x):
        if x.get('sql_tree') is None:
            x['runtime_tree'] = None
            return x

        if x.get('runtime_tree') is None:
            x['runtime_tree'] = build_from_lark_tree(x.get('sql_tree'), add_eps_nodes=True)
        return x

    def _build_ds(filename: str, prefix: str):
        data_file = join(_get_ds_full_dir(ds_tag, split_tag), filename)

        # the read-only redis dataset will only cache the parsed trees,
        # while the volatile memory will store the runtime tree
        ds = VolatileDataset(TransformData(
            dataset=RedisDataset(
                LarkParserDatasetWrapper(
                    grammar_filename=grammar_file,
                    startpoint=_get_grammar_start(grammar_file),
                    parse_keys=['sql'],
                    dataset=IDSupplement(FlattenSeqDS(JsonDataset(data_file), sql_only=sql_only)),
                ),
                conn=conn,
                prefix=prefix,
            ),
            transform_fn=_add_runtime,
        ))
        return ds

    g_tag = _get_grammar_tag_by_filename(grammar_file)
    prefix = f"{'sql' if sql_only else 'qa'}.{ds_tag}.{split_tag}.{g_tag}" + ".{}_"

    train = _build_ds('aligned_train.json', prefix.format('train'))
    dev = _build_ds('aligned_final_dev.json', prefix.format('dev'))
    test = _build_ds('final_test.json', prefix.format('test'))
    logging.info(f"load dataset: {_get_ds_full_dir(ds_tag, split_tag)}")
    return train, dev, test


def install_parsed_sql_datasets(reg: dict = None):
    if reg is None:
        reg = CG_DATA_REG

    for ds_tag, split_tag in product(DATA_PATH.keys(), SPLIT_PATH.keys()):
        for g in GRAMMAR_FILES:
            g_tag = _get_grammar_tag_by_filename(g)
            key = f"sql_{ds_tag}_{split_tag}_{g_tag}"
            reg[key] = partial(_get_parsed_ds, ds_tag, split_tag, g, sql_only=True)
            logging.debug(f"registered {key} lazily")


def install_parsed_qa_datasets(reg: dict = None):
    """
    The obtained instances are
    {
        "sql": str,
        "ext_sql": list of str, possible empty list
        "sent": str,
        "sql_vars": list of {
            "example": str,     // var instance value
            "location": str,    // Whether this occurs in the SQL only, the question only, or both. some cases unknown
            "name": str,    // var instance name (e.g., city0, city1)
            "type": str,    // var type (e.g., city)
        },
        "sent_vars": dict of kv, k is the var instance name, v is the var value.
        // sent vars are assumed to be identical to sql vars, but sometimes they differ.
        // k, v may not be explicitly mentioned in the question, and may not be required for data
    }
    :return: None
    """
    if reg is None:
        reg = CG_DATA_REG

    for ds_tag, split_tag in product(DATA_PATH.keys(), SPLIT_PATH.keys()):
        grammars = []
        run_path = join(SRC_PATH, 'run')
        if osp.exists(run_path):
            grammars += list(join(run_path, f) for f in os.listdir(run_path)
                             if f.endswith('.lark') and f.startswith(ds_tag))

        for g in GRAMMAR_FILES:
            g_tag = _get_grammar_tag_by_filename(g)
            key = f"{ds_tag}_{split_tag}_{g_tag}"
            reg[key] = partial(_get_parsed_ds, ds_tag, split_tag, g, sql_only=False, conn=None)
            logging.debug(f"registered {key} lazily")


def _get_all_ds(keys: list, reg: dict):
    train_dss, dev_dss, test_dss = zip(*[reg[k]() for k in keys])
    train = ChainDataset(train_dss)
    dev = ChainDataset(dev_dss)
    test = ChainDataset(test_dss)
    return train, dev, test


def install_cross_domain_parsed_qa_datasets(reg: dict = None, ds_tags: list = None):
    reg = CG_DATA_REG if reg is None else reg
    ds_tags = ds_tags or DATA_PATH.keys()

    for split_tag in SPLIT_PATH.keys():
        for g in GRAMMAR_FILES:
            g_tag = _get_grammar_tag_by_filename(g)
            chain_keys = [f"qa.{ds_tag}_{split_tag}.{g_tag}" for ds_tag in ds_tags]
            key = f"agsa_{split_tag}_{g_tag}"
            reg[key] = partial(_get_all_ds, keys=chain_keys, reg=reg)
            logging.debug(f"registered {key} lazily")


def install_cross_domain_raw_qa_datasets(reg: dict = None, ds_tags: list = None):
    reg = CG_DATA_REG if reg is None else reg
    ds_tags = ds_tags or DATA_PATH.keys()

    for split_tag in SPLIT_PATH.keys():
        chain_keys = [f"raw_qa.{ds_tag}_{split_tag}" for ds_tag in ds_tags]
        key = f"agsa_{split_tag}"
        reg[key] = partial(_get_all_ds, keys=chain_keys, reg=reg)
        logging.debug(f"registered {key} lazily")

