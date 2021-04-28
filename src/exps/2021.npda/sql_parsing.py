from typing import Literal, List, Dict, Union, Generator, Tuple, Any, Optional, Mapping, Callable, Set
import sys
from os.path import join
sys.path.insert(0, join('..', '..'))
import pickle
from utils.root_finder import find_root
import lark
import logging
logging.basicConfig(level=logging.INFO)
from itertools import chain

from datasets.comp_gen_bundle import install_sql_datasets, Registry
install_sql_datasets()

def parse_dataset(ds_name='advising_cg',
                  grammar_filename: Literal['SQLite.lark', 'MySQL.lark'] = 'SQLite.lark',
                  prefix="",
                  ):
    start = 'parse' if grammar_filename == 'SQLite.lark' else 'query'
    train, dev, test = Registry.get_dataset(ds_name)
    print(f"{ds_name}: train: {len(train)}, dev: {len(dev)}, test: {len(test)} (test will be omitted during extraction)")
    lark_text = join(find_root(), 'src', 'statics', 'grammar', grammar_filename)
    parser = lark.Lark(open(lark_text), start=start, keep_all_tokens=True, )
    trees = []
    for i, example in enumerate(chain(iter(train), iter(dev))):
        sql = example['sql']
        try:
            tree = parser.parse(sql)
        except:
            logging.warning(f"failed to parse example {i}: {sql}")
            continue
        trees.append(tree)
        logging.info(f"example {i} parsed")

    dump_file = f"{ds_name}.{'sqlite' if 'sqlite' in grammar_filename.lower() else 'mysql'}-parse.pkl"
    trees = trees[:len(train)], trees[len(train):]
    pickle.dump(trees, open(prefix + dump_file, "wb"))

def dataset_names():
    names = filter(lambda n: 'cg' in n, Registry._datasets.keys())
    for name in names:
        print('-' * 30)
        print(name)
        yield name

def main():
    # sql parsing is only used for grammar idiom extraction,
    # test set and the question sentences in the training and dev set are thus untouched
    for name in dataset_names():
        parse_dataset(name, prefix='./run/', grammar_filename='MySQL.lark')

if __name__ == '__main__':
    main()
