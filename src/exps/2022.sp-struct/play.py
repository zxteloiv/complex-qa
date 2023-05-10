import os
from itertools import chain

from tqdm import tqdm


from lark import Lark


def main():
    # from shujuji.cogs import install_dataset, cogs_iid, cogs_gen, cogs_iid_parsed, cogs_gen_parsed
    # train, dev, test = cogs_gen_parsed()
    # print(list(map(len, (train, dev, test))))
    # for x in tqdm(chain(train, dev, test)):
    #     print(x.keys(), file=open('a.out', 'w'))
    # from shujuji.smcalflow_cs import smc_by_num, GRAMMAR_FILE
    # train, dev, test = smc_by_num(128)
    # parser = Lark(open(GRAMMAR_FILE), keep_all_tokens=True)
    # print(list(map(len, (train, dev, test))))
    # for x in tqdm(train):
    #     s = x['plan']
    #     tree = parser.parse(s)
    #
    # for x in tqdm(dev):
    #     s = x['plan']
    #     tree = parser.parse(s)
    #
    # for x in tqdm(test):
    #     s = x['plan']
    #     tree = parser.parse(s)
    # from shujuji.compact_cfq import get_ccfq, GRAMMAR_FILE
    # ds = get_ccfq(2)
    # print(list(map(len, ds)))
    # train, dev, test = ds
    # parser = Lark(open(GRAMMAR_FILE), keep_all_tokens=True)
    # for x in tqdm(train):
    #     parser.parse(x['target'])
    # for x in tqdm(dev):
    #     parser.parse(x['target'])
    # for x in tqdm(test):
    #     parser.parse(x['target'])
    import sqlite3
    con = sqlite3.connect('test.db')
    con.execute('create virtual table ftskvmem using fts5(src, tgt);')
    from shujuji.cogs import cogs_iid
    train, dev, test = cogs_iid()
    data = list((x['nl'], x['lf']) for x in train)
    con.executemany('insert into ftskvmem(src, tgt) values (?, ?)', data)
    con.commit()
    con.close()


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    main()
