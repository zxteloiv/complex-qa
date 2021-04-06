import sys
from os.path import join
sys.path.insert(0, join('..', '..'))
from utils.root_finder import find_root
import lark

from datasets.comp_gen_bundle import install_sql_datasets, Registry
install_sql_datasets()

print(f"all_names: {' '.join(Registry._datasets.keys())}")
# for dname, fn in Registry._datasets.items():
#     train, dev, test = fn()
#     for example in iter(train):
#         pass

def parse_sql(ds_name='atis_iid'):
    train, dev, test = Registry.get_dataset(ds_name)
    print(f"{ds_name}: train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

    example = next(iter(test))
    print(example['sql'])

    # from ebnf_compiler import pretty_repr_tree, pretty_derivation_tree
    lark_text = join(find_root(), 'src', 'statics', 'grammar', 'SQLite.lark')
    parser = lark.Lark(open(lark_text), start="parse", keep_all_tokens=True, )
    # lark_text = join(find_root(), 'src', 'statics', 'grammar', 'MySQL.lark')
    # parser = lark.Lark(open(lark_text), start="query", keep_all_tokens=True, )
    from itertools import chain
    for i, example in enumerate(chain(iter(train), iter(dev), iter(test))):
        sql = example['sql']
        try:
            tree = parser.parse(sql)
            print(f"OK {i}")
        except:
            print(sql)
        # print(pretty_repr_tree(tree))
        # print('\n'.join(pretty_derivation_tree(tree)))

def main():
    names = filter(lambda n: 'scholar' not in n and 'iid' in n, Registry._datasets.keys())
    for name in names:
        print('-' * 30)
        print(name)
        parse_sql(name)

if __name__ == '__main__':
    main()
    # ATIS iid: idx: 927 (train)
    # Geo iid: idx: 44 (train)
    # advising:
    # scholar:
