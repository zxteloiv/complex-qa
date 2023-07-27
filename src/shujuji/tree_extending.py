import os
import pickle
import os.path as osp
from functools import partial

from utils.tree import Tree, PreOrderTraverse
from trialbot.data import CompositionalDataset, Dataset


class ExDS(CompositionalDataset):
    def __init__(self, dataset: Dataset, tree_file: str, name_key: str = 'induced_tgt_tree'):
        super().__init__(dataset)
        self.tree_file = tree_file
        self.trees: list[Tree] | None = None
        self.key = name_key

    def __len__(self):
        return len(self.dataset)

    def load_trees(self):
        if self.trees is None:
            self.trees = pickle.load(open(self.tree_file, 'rb'))

    def get_example(self, i: int):
        self.load_trees()
        ex = self.dataset.get_example(i)
        ex[self.key] = self.trees[i]
        return ex


def install_extended_ds(reg: dict = None):
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._datasets

    from .smcalflow_cs import DATA_PATH
    from .cg_bundle import CG_DATA_PATH

    def get_trees(dirname: str) -> dict[str, list[str]]:
        tree_names = ('tree_dump_ds0.pkl', 'tree_dump_ds1.pkl', 'tree_dump_ds2.pkl')
        ts_by_key = {}
        keys = (k for k in os.listdir(dirname) if osp.isdir(osp.join(dirname, k)))
        for k in keys:
            ts_by_key[k] = list(osp.join(dirname, k, f) for f in tree_names)
        return ts_by_key

    smc128_trees = get_trees(osp.join(DATA_PATH, 'source_domain_with_target_num128', 'induced_trees'))
    geocg_trees = get_trees(osp.join(CG_DATA_PATH, 'geography', 'schema_full_split', 'induced_trees'))

    def get_ds(ds_key: str, tree_files: list[str]):
        dss = reg[ds_key]()
        return tuple(map(ExDS, dss, tree_files))

    for k, v in smc128_trees.items():
        reg[f'smc128_tree_{k}'] = partial(get_ds, 'smc128', v)
    for k, v in geocg_trees.items():
        reg[f'geo_cg_tree_{k}'] = partial(get_ds, 'geo_cg', v)
