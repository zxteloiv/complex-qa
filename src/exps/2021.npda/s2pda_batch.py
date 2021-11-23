import logging
import sys, os.path as osp
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))   # up to src
from utils.trialbot.setup import setup_null_argv
import s2pda_qa


def main():
    exp_args = [
        # dict(hparamset="sql_pda_0", dataset="scholar_iid.handcrafted", seed=2021, device=0, translator="cg_sql_pda"),

        # dict(hparamset="sql_pda_1", dataset="scholar_iid.handcrafted", seed=2021, device=0, translator="cg_sql_pda"),
        # dict(hparamset="sql_pda_2", dataset="scholar_iid.handcrafted", seed=2021, device=0, translator="cg_sql_pda"),

        # dict(hparamset="sql_pda_3", dataset="scholar_iid.handcrafted", seed=2021, device=0, translator="cg_sql_pda"),
        # dict(hparamset="sql_pda_4", dataset="scholar_iid.handcrafted", seed=2021, device=0, translator="cg_sql_pda"),

        # dict(hparamset="sql_pda_5", dataset="scholar_iid.handcrafted", seed=2021, device=0, translator="cg_sql_pda"),
        # dict(hparamset="sql_pda_6", dataset="scholar_iid.handcrafted", seed=2021, device=0, translator="cg_sql_pda"),
    ]

    for args in exp_args:
        hp_name = args['hparamset']
        i = hp_name[hp_name.rfind('_') + 1:]
        p = s2pda_qa.Registry.get_hparamset(hp_name)

        if p.emb_sz != p.hidden_sz:
            continue

        logfile_name = f'exp_npda_sch_iid_{i}_elayer{p.num_enc_layers}_enc{p.enc_sz}_emb256_hid256.log'
        logging.basicConfig(filename=logfile_name, force=True)
        s2pda_qa.make_trialbot(setup_null_argv(memo=f"gs{i}", **args), print_details=True, epoch_eval=True).run()


if __name__ == '__main__':
    main()
