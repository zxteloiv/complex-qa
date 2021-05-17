# ##
# The bundle of classic datasets re-split for compositional generalization evaluation.
# introduced by the paper Oren et al. 2020.
# Improving Compositional Generalization in Semantic Parsing
# https://www.aclweb.org/anthology/2020.findings-emnlp.225/
# https://github.com/inbaroren/improving-compgen-in-semparse
# ##

from trialbot.data import JsonDataset
from .composition_dataset import CompositionalDataset
from .redis_dataset import RedisDataset
from .lark_dataset import LarkParserDatasetWrapper
from trialbot.utils.root_finder import find_root
import os
from os.path import join
from functools import partial
import logging

ROOT = find_root()
CG_DATA_REG = dict()

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
                'sent_vars': group['sentences'][idx[1]]['variables']
            }
        else:
            instance = {
                "sql": group['sql'][idx[1]],
                'variables': group['variables'],
            }
        return instance

def _get_sql_ds(data_name: str, *, use_iid: bool):
    ds_dir = join(ROOT, 'data', 'CompGen', 'sql data', data_name,
                  'new_question_split' if use_iid else 'schema_full_split')
    train = FlattenSeqDS(JsonDataset(join(ds_dir, 'aligned_train.json')), sql_only=True)
    dev = FlattenSeqDS(JsonDataset(join(ds_dir, 'aligned_final_dev.json')), sql_only=True)
    test = FlattenSeqDS(JsonDataset(join(ds_dir, 'final_test.json')), sql_only=True)
    print(f"load dataset: {ds_dir}")
    return train, dev, test

def install_sql_datasets(reg: dict = None):
    if reg is None:
        reg = CG_DATA_REG
    domains = ["atis", "geo", "advising", "scholar"]
    path_names = ["atis", "geography", "advising", "scholar"]
    for domain, pathname in zip(domains, path_names):
        name = f"{domain}_iid"
        reg[name] = partial(_get_sql_ds, pathname, use_iid=True)
        name = f"{domain}_cg"
        reg[name] = partial(_get_sql_ds, pathname, use_iid=False)

def _get_qa_ds(data_name: str, *, use_iid: bool, grammar_file: str, sql_only: bool):
    ds_dir = join(ROOT, 'data', 'CompGen', 'sql data', data_name,
                  'new_question_split' if use_iid else 'schema_full_split')
    grammar_tag = grammar_file[grammar_file.rfind('/') + 1:grammar_file.index('.lark')]
    grammar_tag = grammar_tag[grammar_tag.rfind('_') + 1:]
    iid_tag = 'iid' if use_iid else 'cg'

    def _build_ds(filename: str, split_tag: str):
        nonlocal ds_dir, grammar_tag
        ds = RedisDataset(
            dataset=LarkParserDatasetWrapper(
                grammar_filename=grammar_file,
                startpoint='parse' if 'sqlite' in grammar_file.lower() else 'query',
                parse_keys=['sql'],
                dataset=FlattenSeqDS(JsonDataset(join(ds_dir, filename)), sql_only=sql_only)
            ),
            conn=('localhost', 6379, 2),
            prefix=f"{split_tag}.{iid_tag}.{data_name}.{grammar_tag}_",
        )
        return ds

    train = _build_ds('aligned_train.json', 'train')
    dev = _build_ds('aligned_final_dev.json', 'dev')
    test = _build_ds('final_test.json', 'test')
    print(f"load dataset: {ds_dir}")
    return train, dev, test

def install_qa_datasets(reg: dict = None):
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
    domains = ["atis", "geo", "advising", "scholar"]
    path_names = ["atis", "geography", "advising", "scholar"]
    grammar_path = join('..', '..', 'statics', 'grammar')

    if reg is None:
        reg = CG_DATA_REG

    for domain, pathname in zip(domains, path_names):
        grammars = [join(grammar_path, 'MySQL.lark'), join(grammar_path, 'SQLite.lark')]
        for g in grammars:
            tag = g[g.rfind('/') + 1:g.index('.lark')].lower()
            reg[f"{domain}_iid.{tag}"] = partial(_get_qa_ds, pathname, use_iid=True, sql_only=False, grammar_file=g)
            logging.debug(f"registered {domain}_iid.{tag} lazily")
            reg[f"{domain}_cg.{tag}"] = partial(_get_qa_ds, pathname, use_iid=False, sql_only=False, grammar_file=g)
            logging.debug(f"registered {domain}_cg.{tag} lazily")

def install_sql_qa_datasets(reg: dict = None):
    if reg is None:
        reg = CG_DATA_REG
    domains = ["atis", "geo", "advising", "scholar"]
    path_names = ["atis", "geography", "advising", "scholar"]
    for domain, path_name in zip(domains, path_names):
        grammars = list(join('run', f) for f in os.listdir('./run') if f.endswith('.lark') and f.startswith(domain))
        for g in grammars:
            tag = g[g.rfind('/') + 1:g.index('.lark')].lower()
            reg[domain + '_iid.' + tag] = partial(_get_qa_ds, path_name, use_iid=True, grammar_file=g, sql_only=False)
            logging.debug(f"registered {domain}_iid.{tag} lazily")
            reg[domain + '_cg.' + tag] = partial(_get_qa_ds, path_name, use_iid=False, grammar_file=g, sql_only=False)
            logging.debug(f"registered {domain}_cg.{tag} lazily")

