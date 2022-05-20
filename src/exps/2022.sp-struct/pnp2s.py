from trialbot.training import Registry
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.root_finder import find_root
import sys
sys.path.insert(0, find_root('.SRC'))


def main():
    from utils.trialbot.setup_cli import setup as setup_cli
    from libs2s import setup_common_bot
    import datasets.comp_gen_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    import datasets.cg_bundle_translator

    bot = setup_common_bot(args=setup_cli(translator='s2s', seed=2021, device=0))
    bot.run()


@Registry.hparamset()
def sch_tranx():
    from libs2s import base_hparams
    p = base_hparams()
    # only list some crucial changes here
    p.src_namespace = 'sent'
    p.tgt_namespace = 'rule_seq'
    p.batch_sz = 16
    p.TRAINING_LIMIT = 400
    p.enc_out_dim = 150
    p.encoder = 'perturb_parse'
    p.num_enc_layers = 2

    # the best param from grid search under the single seed 2021, on both tranx and s2s models;
    p.decoder_init_strategy = "avg_all"
    p.enc_dec_trans_usage = 'consistent'
    p.enc_attn = "dot_product"
    return p


@Registry.hparamset()
def sch_s2s():
    p = sch_tranx()
    p.tgt_namespace = 'sql'
    return p


if __name__ == '__main__':
    main()

