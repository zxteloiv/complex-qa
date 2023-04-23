from itertools import chain


def main():
    from shujuji.cogs import install_dataset, cogs_iid, cogs_gen, cogs_iid_parsed, cogs_gen_parsed
    from lark import Lark
    train, dev, test = cogs_gen_parsed()
    print(list(map(len, (train, dev, test))))
    from tqdm import tqdm
    for x in tqdm(chain(train, dev, test)):
        print(x.keys(), file=open('a.out', 'w'))


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    main()
