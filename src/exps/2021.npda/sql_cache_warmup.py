import sys
sys.path.insert(0, '../..')
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_sql_qa_datasets()
cg_bundle.install_parsed_sql_datasets()

def get_dataset():
    tags = [
        # "scholar_iid.scholar_cg.sqlite.20",
        # "scholar_cg.scholar_cg.sqlite.20",
        # "scholar_iid.scholar_cg.mysql.30",
        # "scholar_cg.scholar_cg.mysql.30",
        # "atis_iid.atis_cg.sqlite-parse.30",
        # "atis_cg.atis_cg.sqlite-parse.30",
        # "atis_iid.atis_cg.mysql-parse.50",
        # "atis_cg.atis_cg.mysql-parse.50",
        # "advising_iid.advising_cg.sqlite-parse.30",
        # "advising_cg.advising_cg.sqlite-parse.30",
        # "advising_iid.advising_cg.mysql-parse.50",
        # "advising_cg.advising_cg.mysql-parse.50",
        # "atis_iid.sqlite",
        # "atis_cg.sqlite",
        # "geo_iid.sqlite",
        # "geo_cg.sqlite",
        # "advising_iid.sqlite",
        # "advising_cg.sqlite",
        # "scholar_iid.sqlite",
        # "scholar_cg.sqlite",
        # "atis_iid.mysql",
        # "atis_cg.mysql",
        # "geo_iid.mysql",
        # "geo_cg.mysql",
        # "advising_iid.mysql",
        # "advising_cg.mysql",
        # "scholar_iid.mysql",
        # "scholar_cg.mysql",
        # "pure_sql.scholar_cg.sqlite",
        # "pure_sql.scholar_cg.mysql",
        # "pure_sql.atis_cg.sqlite",
        # "pure_sql.geo_cg.sqlite",
        # "pure_sql.advising_cg.sqlite",
        # "pure_sql.atis_cg.mysql",
        # "pure_sql.geo_cg.mysql",
        # "pure_sql.advising_cg.mysql",
        "pure_sql.atis_iid.sqlite",
        "pure_sql.geo_iid.sqlite",
        "pure_sql.advising_iid.sqlite",
        "pure_sql.scholar_iid.sqlite",
        "pure_sql.atis_iid.mysql",
        "pure_sql.geo_iid.mysql",
        "pure_sql.advising_iid.mysql",
        "pure_sql.scholar_iid.mysql",
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