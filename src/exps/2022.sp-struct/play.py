import os
from itertools import chain

from tqdm import tqdm


def main():
    # from shujuji.cogs import install_dataset, cogs_iid, cogs_gen, cogs_iid_parsed, cogs_gen_parsed
    # from lark import Lark
    # train, dev, test = cogs_gen_parsed()
    # print(list(map(len, (train, dev, test))))
    # for x in tqdm(chain(train, dev, test)):
    #     print(x.keys(), file=open('a.out', 'w'))
    from shujuji.smcalflow_cs import smc_by_num, GRAMMAR_FILE
    train, dev, test = smc_by_num(128)
    from lark import Lark
    parser = Lark(open(GRAMMAR_FILE), keep_all_tokens=True)
    print(list(map(len, (train, dev, test))))
    for x in tqdm(train):
        s = x['plan']
        tree = parser.parse(s)

    for x in tqdm(dev):
        s = x['plan']
        tree = parser.parse(s)

    for x in tqdm(test):
        s = x['plan']
        tree = parser.parse(s)


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    main()
