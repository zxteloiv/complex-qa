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
    install_hparamsets()

    args = setup_cli(seed=2021, device=0)
    args.translator = guess_translator(args.hparamset)
    bot = setup_common_bot(args=args)

    training_limit = 30 if 'atis' in args.dataset or 'advising' in args.dataset else 0
    bot.run(training_limit)     # override the hparamset.TRAINING_LIMIT config


def encoder_decorators():
    def seq(p):
        p.TRAINING_LIMIT = 400
        p.batch_sz = 16
        p.src_namespace = 'sent'
        p.decoder_init_strategy = "forward_last_parallel"
        p.encoder = 'bilstm'
        return p

    def onlstm(p):
        p.TRAINING_LIMIT = 400
        p = seq(p)
        p.encoder = 'onlstm'
        p.use_cell_based_encoder = True
        p.cell_encoder_is_bidirectional = True
        return p

    def diora(p):
        p.TRAINING_LIMIT = 400
        p.src_namespace = 'sent'
        p.enc_out_dim = 150
        p.encoder = 'diora'

        p.decoder_init_strategy = "avg_all"
        p.diora_concat_outside = True
        p.diora_loss_enabled = True
        p.diora_topk = 2
        return p

    def pnp(p):
        p.TRAINING_LIMIT = 300
        p.enc_out_dim = 150
        p.src_namespace = 'sent'
        p.encoder = 'perturb_parse'
        p.num_enc_layers = 2
        p.decoder_init_strategy = "avg_all"
        return p

    def small_tdpcfg(p):
        p.src_namespace = 'sent'
        p.compound_encoder = 'tdpcfg'
        p.num_pcfg_nt = 10
        p.num_pcfg_pt = 20
        p.td_pcfg_rank = 10
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

    def small_rtdpcfg(p):
        p = small_tdpcfg(p)
        p.compound_encoder = 'reduced_tdpcfg'
        return p

    def small_cpcfg(p):
        p = small_tdpcfg(p)
        p.compound_encoder = 'cpcfg'
        return p

    def small_rcpcfg(p):
        p = small_tdpcfg(p)
        p.compound_encoder = 'reduced_cpcfg'
        return p

    def syn(p):
        p.src_namespace = 'sent'
        p.encoder = 'syn_gcn'
        p.decoder_init_strategy = "avg_all"
        p.syn_gcn_activation = 'mish'
        p.num_enc_layers = 2
        p.enc_out_dim = 200
        return p

    encoders = {}

    def _add_encoder(func):
        encoders[func.__name__] = func

    _add_encoder(seq)
    _add_encoder(onlstm)
    _add_encoder(diora)
    _add_encoder(pnp)
    _add_encoder(small_tdpcfg)
    _add_encoder(small_rtdpcfg)
    _add_encoder(small_cpcfg)
    _add_encoder(small_rcpcfg)
    _add_encoder(syn)

    def plm(p):
        p.src_namespace = None
        p.decoder_init_strategy = "avg_all"
        p.lr_scheduler_kwargs = None
        return p

    def electra(p):
        p = plm(p)
        p.TRANSLATOR_KWARGS = {"model_name": "google/electra-base-discriminator"}
        p.encoder = 'plm:google/electra-base-discriminator'
        return p

    def bert(p):
        p = plm(p)
        p.TRANSLATOR_KWARGS = {"model_name": "bert-base-uncased"}
        p.encoder = 'plm:bert-base-uncased'
        return p

    _add_encoder(electra)
    _add_encoder(bert)
    return encoders


def decoder_decorators():
    def seq(p):
        p.tgt_namespace = 'sql'
        return p

    def onlstm(p):
        p.tgt_namespace = 'sql'
        p.decoder = 'onlstm'
        return p

    def tranx(p):
        p.tgt_namespace = 'rule_seq'
        return p

    decoders = {
        'seq': seq,
        'onlstm': onlstm,
        'tranx': tranx
    }
    return decoders


def guess_translator(pname: str) -> str:
    tranx_dec = pname.endswith('tranx')
    syn_parse_enc = pname.startswith('syn')
    plm_enc = any(ename in pname for ename in ('electra', 'bert'))

    if tranx_dec and syn_parse_enc:
        return 'syn2tranx'
    elif tranx_dec and plm_enc:
        return 'plm2tranx'
    elif syn_parse_enc:
        return 'syn2s'
    elif plm_enc:
        return 'plm2s'
    elif tranx_dec:
        return 'tranx'
    else:
        return 's2s'


def _compose_hp_func(funcname, efunc, dfunc):
    from libs2s import base_hparams

    def _func():
        return dfunc(efunc(base_hparams()))

    setattr(_func, '__name__', funcname)
    setattr(_func, '__qualname__', funcname)
    return _func


def install_hparamsets():
    encoders = encoder_decorators()
    decoders = decoder_decorators()

    for ename, efunc in encoders.items():
        for dname, dfunc in decoders.items():
            hp_name = f'{ename}2{dname}'
            Registry._hparamsets[hp_name] = _compose_hp_func(hp_name, efunc, dfunc)


if __name__ == '__main__':
    main()
