

def main():
    from shujuji.cogs import install_dataset, cogs_iid, cogs_gen
    from lark import Lark
    parser = Lark(open('../../statics/grammar/cogs.lark'), keep_all_tokens=True)
    train, dev, test = cogs_iid()
    train2, dev2, test2 = cogs_gen()
    print(list(map(len, cogs_iid())))
    print(list(map(len, cogs_gen())))

    from itertools import chain
    for i, x in enumerate(chain(train, dev, test, train2, dev2, test2)):
        nl, lf, tag = x[:3]
        # if tag == 'primitive':
        #     continue
        tree = parser.parse(lf)
    print('done!')


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    main()
