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
def sch_tdpcfg2s():
    from libs2s import base_hparams
    p = base_hparams()
    p.batch_sz = 16
    p.src_namespace = 'sent'
    p.tgt_namespace = 'sql'
    p.encoder = 'bilstm'
    p.compound_encoder = 'tdpcfg'
    p.num_pcfg_nt = 20
    p.num_pcfg_pt = 40
    p.td_pcfg_rank = 20

    p.emb_sz = 100
    p.hidden_sz = 200
    p.pcfg_hidden_dim = p.hidden_sz
    p.pcfg_encoding_dim = p.hidden_sz
    p.enc_out_dim = p.hidden_sz
    p.dec_in_dim = p.hidden_sz
    p.dec_out_dim = p.hidden_sz
    p.proj_in_dim = p.emb_sz

    p.decoder_init_strategy = "avg_all"
    p.enc_attn = 'dot_product'

    return p

@Registry.hparamset()
def sch_tdpcfg2s_small():
    p = sch_tdpcfg2s()
    p.num_pcfg_nt = 10
    p.num_pcfg_pt = 20
    p.td_pcfg_rank = 10
    return p

@Registry.hparamset()
def sch_tdpcfg2s_big():
    p = sch_tdpcfg2s()
    p.num_pcfg_nt = 30
    p.num_pcfg_pt = 60
    p.td_pcfg_rank = 30
    return p

@Registry.hparamset()
def sch_tdpcfg2tranx():
    p = sch_tdpcfg2s()
    p.tgt_namespace = 'rule_seq'
    return p

@Registry.hparamset()
def sch_tdpcfg2tranx_small():
    p = sch_tdpcfg2s_small()
    p.tgt_namespace = 'rule_seq'
    return p

@Registry.hparamset()
def sch_tdpcfg2tranx_big():
    p = sch_tdpcfg2s_big()
    p.tgt_namespace = 'rule_seq'
    return p

@Registry.hparamset()
def sch_cpcfg2s():
    p = sch_tdpcfg2s()
    p.num_pcfg_nt = 20
    p.num_pcfg_pt = 40
    p.compound_encoder = 'cpcfg'
    return p

@Registry.hparamset()
def sch_cpcfg2tranx():
    p = sch_cpcfg2s()
    p.tgt_namespace = 'rule_seq'
    return p

@Registry.hparamset()
def sch_rcpcfg2s():
    p = sch_tdpcfg2s()
    p.compound_encoder = 'reduced_cpcfg'
    return p

@Registry.hparamset()
def sch_rcpcfg2tranx():
    p = sch_tdpcfg2tranx()
    p.compound_encoder = 'reduced_cpcfg'
    return p

@Registry.hparamset()
def sch_rcpcfg2s_small():
    p = sch_tdpcfg2s_small()
    p.compound_encoder = 'reduced_cpcfg'
    return p

@Registry.hparamset()
def sch_rcpcfg2tranx_small():
    p = sch_tdpcfg2tranx_small()
    p.compound_encoder = 'reduced_cpcfg'
    return p

@Registry.hparamset()
def sch_rcpcfg2s_big():
    p = sch_tdpcfg2s_big()
    p.compound_encoder = 'reduced_cpcfg'
    return p

@Registry.hparamset()
def sch_rcpcfg2tranx_big():
    p = sch_tdpcfg2tranx_big()
    p.compound_encoder = 'reduced_cpcfg'
    return p

@Registry.hparamset()
def sch_rtdpcfg2s():
    p = sch_tdpcfg2s()
    p.compound_encoder = 'reduced_tdpcfg'
    return p

@Registry.hparamset()
def sch_rtdpcfg2tranx():
    p = sch_tdpcfg2tranx()
    p.compound_encoder = 'reduced_tdpcfg'
    return p

@Registry.hparamset()
def sch_rtdpcfg2s_small():
    p = sch_tdpcfg2s_small()
    p.compound_encoder = 'reduced_tdpcfg'
    return p

@Registry.hparamset()
def sch_rtdpcfg2tranx_small():
    p = sch_tdpcfg2tranx_small()
    p.compound_encoder = 'reduced_tdpcfg'
    return p

@Registry.hparamset()
def sch_rtdpcfg2s_big():
    p = sch_tdpcfg2s_big()
    p.compound_encoder = 'reduced_tdpcfg'
    return p

@Registry.hparamset()
def sch_rtdpcfg2tranx_big():
    p = sch_tdpcfg2tranx_big()
    p.compound_encoder = 'reduced_tdpcfg'
    return p

if __name__ == '__main__':
    main()
