import os
import pickle
import sqlite3
from difflib import SequenceMatcher
from os import path as osp

from .cx import Construction
from .pattern_find_utils import (
generalize_cxs,
)


class Inventory:
    def __init__(self):
        self.training: bool = True  # in training mode, inventory will keep update online
        self.db = sqlite3.connect(':memory:')
        self.db.execute("""create virtual table cxs using fts5(cxid, form, meaning, tag);""")
        self.cxs: list[Construction] = []
        self.limit: int = 5

    def __call__(self, nl: list[str], lf: list[str]):
        for x, y in zip(nl, lf):
            sem_net = self.parse(x)
            if sem_net is None:
                self.learn(x, y)
            else:
                self.update(sem_net)
        return {}

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        target = sqlite3.connect(osp.join(dirname, 'index.db'))
        self.db.backup(target)
        pickle.dump(self.cxs, open(osp.join(dirname, 'cxs.pkl'), 'wb'))
        return self

    def load(self, dirname):
        source = sqlite3.connect(osp.join(dirname, 'index.db'))
        source.backup(self.db)
        self.cxs = pickle.load(open(os.path.join(dirname, 'cxs.pkl'), 'rb'))
        return self

    def insert_cx(self, cx: Construction) -> int:
        cxid = len(self.cxs)
        self.cxs.append(cx)
        self.db.execute("""insert into cxs values (?, ?, ?, ?)""", (cxid, cx.form, cx.meaning, cx.tag))
        self.db.commit()
        return cxid

    def find_cxs(self, x, y) -> list[Construction]:
        # find a holophrase cx
        fts_query = '"{0}"'.format(x)
        results = self.db.execute("select cxid from cxs where form match ? "
                                  "and tag = 'holophrase' order by rank limit ?",
                                  (fts_query, self.limit)).fetchall()
        return [self.cxs[cxid[0]] for cxid in results]

    def learn(self, x, y):
        # self.mech1(x, y)
        self.mech2(x, y)
        pass

    def mech1(self, x, y):
        # save a holophrase cx
        cxid = self.insert_cx(Construction.deserialize(x, y, tag='holophrase'))
        return cxid

    def mech2(self, x, y):
        cxs = self.find_cxs(x, y)
        for cx in cxs:
            if cx.form == x:
                continue
            try:
                item_cx = generalize_cxs(x, y, cx.form, cx.meaning)
                self.insert_cx(item_cx)
            except ValueError:
                pass

    def update(self, sem_net):
        pass

    def parse(self, x):
        # if len(self.cxs) == 0:
        #     return None     # impossible to parse
        return None
