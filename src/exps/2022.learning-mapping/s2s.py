from trialbot.training import Registry
from os import path as osp


def main():
    from utils.trialbot.setup_cli import setup as setup_cli
    import shujuji.cg_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    cg_bundle.install_cross_domain_parsed_qa_datasets(Registry._datasets)
    cg_bundle.install_raw_qa_datasets(Registry._datasets)
    cg_bundle.install_cross_domain_raw_qa_datasets(Registry._datasets)
    from utils.s2s_arch.setup_bot import setup_common_bot
    bot = setup_common_bot(setup_cli(seed=2021, device=0, translator='s2s', dataset='raw_qa.all_iid', epoch=40))
    bot.run()


@Registry.hparamset()
def s2s():
    from models.base_s2s.model_factory import Seq2SeqBuilder
    p = Seq2SeqBuilder.base_hparams()
    p.TRAINING_LIMIT = 400
    p.batch_sz = 16
    p.src_namespace = 'sent'
    p.tgt_namespace = 'sql'
    p.encoder = 'bilstm'
    return p


@Registry.hparamset()
def plm2s():
    p = s2s()
    p.lr_scheduler_kwargs = None
    p.src_namespace = None
    p.decoder_init_strategy = "avg_all"
    plm_path = osp.abspath(osp.expanduser('~/.cache/complex_qa/bert-base-uncased'))
    p.encoder = 'plm:' + plm_path
    p.TRANSLATOR_KWARGS = {"model_name": plm_path}
    return p


@Registry.hparamset()
def hungarian_reg():
    p = s2s()
    p.attn_supervision = 'hungarian_reg'
    return p


@Registry.hparamset()
def hungarian_xent():
    p = s2s()
    p.attn_supervision = 'hungarian_xent'
    return p


@Registry.hparamset()
def hungarian_reg_xent():
    p = s2s()
    p.attn_supervision = 'hungarian_reg_xent'
    return p


@Registry.hparamset()
def rev_hungarian_xent():
    p = s2s()
    p.attn_supervision = 'rev_hungarian_xent'
    return p


@Registry.hparamset()
def rev_hungarian_reg_xent():
    p = s2s()
    p.attn_supervision = 'rev_hungarian_reg_xent'
    return p


@Registry.hparamset()
def plm2s_hungarian_reg():
    p = plm2s()
    p.attn_supervision = 'hungarian_reg'
    return p


@Registry.hparamset()
def plm2s_hungarian_sup():
    p = plm2s()
    p.attn_supervision = 'hungarian_sup'
    return p


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    main()
