import sys
sys.path.insert(0, '../..')
import datasets.cfq as cfq

def get_dataset():
    ds_functions = [
        cfq.cfq_mcd1_simplified,
        cfq.cfq_mcd2_simplified,
        cfq.cfq_mcd3_simplified,
    ]
    for ds_f in ds_functions:
        ds_tag = ds_f.__name__
        print(ds_tag)
        train, dev, test = ds_f()
        yield ds_tag + "_train", train
        yield ds_tag + "_dev", dev
        yield ds_tag + "_test", test

def build_sql_vocab(pid, total):
    print(pid, total)
    for tag, ds in get_dataset():
        # ds: IndexDataset/RedisDataset/LarkParserDatasetWrapper/PickleDataset
        for i in range(len(ds)):
            if i % total != pid:
                continue

            print(f"read {i} examples from {tag}")
            _ = ds[i]

if __name__ == '__main__':
    pid, total = int(sys.argv[1]), int(sys.argv[2])
    build_sql_vocab(pid, total)
