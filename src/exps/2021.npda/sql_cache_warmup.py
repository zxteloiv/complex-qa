import sys
sys.path.insert(0, '../..')
import shujuji.cg_bundle as cg_bundle
cg_bundle.install_parsed_qa_datasets()
cg_bundle.install_parsed_sql_datasets()

def get_dataset():
    tags = [
        # currently only the idiom-extraction will need the warmup,
        # because the exported grammar size is small and can be used at runtime
        "atis_iid.handcrafted",
        "geo_iid.handcrafted",
        "advising_iid.handcrafted",
        "scholar_iid.handcrafted",
        "scholar_cg.handcrafted",
        "atis_cg.handcrafted",
        "geo_cg.handcrafted",
        "advising_cg.handcrafted",
    ]
    for ds_tag in tags:
        print(ds_tag)
        train, dev, test = cg_bundle.CG_DATA_REG[ds_tag]()
        yield ds_tag + "_train", train
        yield ds_tag + "_dev", dev
        yield ds_tag + "_test", test

def build_sql_vocab(pid, total):
    print(pid, total)
    for tag, ds in get_dataset():
        # ds: RedisDataset(LarkParserDatasetWrapper(FlattenSeqDS))
        for i in range(len(ds.dataset.dataset)):
            if i % total != pid:
                continue

            print(f"read {i} examples from {tag}")
            _ = ds[i]

        _ = len(ds)

if __name__ == '__main__':
    pid, total = int(sys.argv[1]), int(sys.argv[2])
    build_sql_vocab(pid, total)