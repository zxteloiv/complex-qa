import logging
logging.basicConfig(level=logging.DEBUG)
import sys
from os.path import join
sys.path.insert(0, join('..', '..'))
import lark
TREE, TOKEN = lark.Tree, lark.Token
import utils.cfg as cfg
import shujuji.comp_gen_bundle as cg_bundle
from idioms.miner import GreedyIdiomMiner
from idioms.export_conf import get_export_conf


def sql_data_mining(prefix=""):
    for ds_name, get_ds_fn in cg_bundle.CG_DATA_REG.items():
        logging.info(f"================== {ds_name} ====================")
        train, dev, test = get_ds_fn()
        print(f"{ds_name}: train: {len(train)}, dev: {len(dev)}, test: {len(test)}")
        train_trees = list(filter(None, (x['sql_tree'] for x in train)))
        dev_trees = list(filter(None, (x['sql_tree'] for x in dev)))
        print(f"{ds_name}: initial grammar success ratio: "
              f"{len(train_trees)} / {len(train)} for training, "
              f"{len(dev_trees)} / {len(dev)} for testing.")
        miner = GreedyIdiomMiner(train_trees, dev_trees, ds_name[ds_name.index('pure_sql.') + 9:],
                                 max_mining_steps=200,
                                 data_prefix=prefix,
                                 freq_lower_bound=1,
                                 retain_recursion=True,
                                 )
        miner.mine()
        miner.evaluation()
        lex_file, start, export_terminals, excluded = get_export_conf(ds_name.lower())
        for i in range(0, len(miner.stat_by_iter), 5):
            miner.export_kth_rules(i, lex_file, start, export_terminals=export_terminals, excluded_terminals=excluded)


def cfq_dataset_mining():
    import shujuji.cfq as cfq_data
    from tqdm import tqdm
    train, dev, test = cfq_data.cfq_mcd1_classic()
    print("loading training trees...")
    train_tree = [obj['sparqlPatternModEntities_tree'] for obj in tqdm(train)]
    print("loading dev trees...")
    dev_tree = [obj['sparqlPatternModEntities_tree'] for obj in tqdm(dev)]
    miner = GreedyIdiomMiner(train_tree, dev_tree, 'cfq_mcd1',
                             freq_lower_bound=3,
                             data_prefix='run/',
                             sample_percentage=.2,
                             max_mining_steps=200,
                             retain_recursion=True,
                             )
    miner.mine()
    miner.evaluation()
    lex_file, start, export_terminals, excluded = get_export_conf('cfq_mcd1')
    for i in range(0, len(miner.stat_by_iter), 10):
        miner.export_kth_rules(i, lex_file, start, export_terminals=export_terminals, excluded_terminals=excluded)


def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'cfq':
        logging.info('run mining for cfq...')
        # sparql part
        cfq_dataset_mining()
    else:
        logging.info('run mining for sql...')
        # sql part
        cg_bundle.install_parsed_sql_datasets()

        sql_data_mining(prefix='./run/')
        # sql_load_miner_state(prefix='./run/')

if __name__ == '__main__':
    main()
