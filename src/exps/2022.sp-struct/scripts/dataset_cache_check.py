import redis
from trialbot.utils.root_finder import find_root
import sys
sys.path.insert(0, find_root('.SRC'))


def main():
    import shujuji.cg_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets()
    reg = cg_bundle.CG_DATA_REG
    for k, v in reg.items():
        print('dataset reg key:', k)
        train, dev, test = v()
        # do any thing you want. e.g. len(train), len(train[0]['sent'].split())

    pass


if __name__ == '__main__':
    main()
