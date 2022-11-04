from trialbot.training import Registry


def main():
    from utils.trialbot.setup_cli import setup as setup_cli
    from libs2s import setup_common_bot
    import datasets.comp_gen_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    import datasets.cg_bundle_translator
    bot = setup_common_bot(setup_cli(seed=2021, device=0, translator='s2s'))
    bot.run()


@Registry.hparamset()
def s2s():
    from libs2s import base_hparams
    p = base_hparams()
    p.TRAINING_LIMIT = 400
    p.batch_sz = 16
    p.src_namespace = 'sent'
    p.tgt_namespace = 'sql'
    p.decoder_init_strategy = "forward_last_parallel"
    p.encoder = 'bilstm'
    return p


@Registry.hparamset()
def hungarian_reg():
    p = s2s()
    p.attn_supervision = 'hungarian_reg'
    return p


@Registry.hparamset()
def hungarian_sup():
    p = s2s()
    p.attn_supervision = 'hungarian_sup'
    return p


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    main()
