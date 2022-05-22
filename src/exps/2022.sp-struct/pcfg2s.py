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
    bot = setup_common_bot(setup_cli(translator='s2s', seed=2021, device=0))
    bot.run()


@Registry.hparamset()
def sch_s2s():
    from libs2s import base_hparams
    p = base_hparams()
    p.batch_sz = 1
    p.src_namespace = 'sent'
    p.tgt_namespace = 'sql'
    p.encoder = 'bilstm'
    p.compound_encoder = 'cpcfg'
    p.num_pcfg_nt = 30
    p.num_pcfg_pt = 60

    p.emb_sz = 100
    p.hidden_sz = 200
    p.pcfg_hidden_dim = p.hidden_sz
    p.pcfg_encoding_dim = p.hidden_sz
    p.enc_out_dim = p.hidden_sz
    p.dec_in_dim = p.hidden_sz
    p.dec_out_dim = p.hidden_sz
    p.proj_in_dim = p.emb_sz

    return p


if __name__ == '__main__':
    main()
