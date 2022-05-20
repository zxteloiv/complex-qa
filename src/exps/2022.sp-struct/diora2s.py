# mostly a seq2seq but the encoder is modified for the Diora or S-Diora model
from trialbot.utils.grid_search_helper import import_grid_search_parameters
from trialbot.training import Registry
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
    p.batch_sz = 16
    p.TRAINING_LIMIT = 400
    p.enc_out_dim = 150
    p.src_namespace = 'sent'
    p.tgt_namespace = 'rule_seq'

    p.encoder = 'diora'

    # the best param from grid search under the single seed 2021, on both tranx and s2s models;
    p.decoder_init_strategy = "avg_all"
    p.diora_concat_outside = True
    p.diora_loss_enabled = True
    p.diora_topk = 2

    return p


@Registry.hparamset()
def sch_s2s():
    p = sch_tranx()
    p.tgt_namespace = 'sql'
    return p


if __name__ == '__main__':
    main()

