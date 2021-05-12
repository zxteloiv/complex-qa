import sys
sys.path.insert(0, '../..')
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_sql_qa_datasets()

def get_dataset():
    tags = [
        "atis_iid.atis_cg.sqlite-parse.30",
        "atis_cg.atis_cg.sqlite-parse.30",
        "atis_iid.atis_cg.sqlite-parse.50",
        "atis_cg.atis_cg.sqlite-parse.50",
        "atis_iid.atis_cg.mysql-parse.70",
        "atis_cg.atis_cg.mysql-parse.70",
        "advising_iid.advising_cg.sqlite-parse.30",
        "advising_cg.advising_cg.sqlite-parse.30",
        "advising_iid.advising_cg.mysql-parse.50",
        "advising_cg.advising_cg.mysql-parse.50",
        "advising_iid.advising_cg.sqlite-parse.40",
        "advising_cg.advising_cg.sqlite-parse.40",
        "advising_iid.advising_cg.mysql-parse.60",
        "advising_cg.advising_cg.mysql-parse.60",
        "scholar_iid.scholar_cg.sqlite-parse.30",
        "scholar_cg.scholar_cg.sqlite-parse.30",
        "scholar_iid.scholar_cg.mysql-parse.40",
        "scholar_cg.scholar_cg.mysql-parse.40",
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

if __name__ == '__main__':
    pid, total = int(sys.argv[1]), int(sys.argv[2])
    build_sql_vocab(pid, total)