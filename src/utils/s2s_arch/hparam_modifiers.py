from functools import wraps
from typing import Callable, TypeVar, Dict, List, Any, Union, Collection

T = TypeVar('T')
MODIFIER = Callable[[T], T]
MOD_DICT = Dict[str, MODIFIER]


def encoder_modifiers() -> MOD_DICT:
    def seq(p):
        p.decoder_init_strategy = "forward_last_all"
        p.encoder = 'bilstm'
        return p

    def onlstm(p):
        p = seq(p)
        p.encoder = 'onlstm'
        p.use_cell_based_encoder = True
        p.cell_encoder_is_bidirectional = True
        return p

    def diora(p):
        p.enc_out_dim = 150
        p.encoder = 'diora'

        p.decoder_init_strategy = "avg_all"
        p.diora_concat_outside = True
        p.diora_loss_enabled = True
        p.diora_topk = 2
        return p

    def pnp(p):
        p.enc_out_dim = 150
        p.encoder = 'perturb_parse'
        p.num_enc_layers = 2
        p.decoder_init_strategy = "avg_all"
        return p

    def tdpcfg(p):
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

    def rtdpcfg(p):
        p = tdpcfg(p)
        p.compound_encoder = 'reduced_tdpcfg'
        return p

    def cpcfg(p):
        p = tdpcfg(p)
        p.compound_encoder = 'cpcfg'
        return p

    def rcpcfg(p):
        p = tdpcfg(p)
        p.compound_encoder = 'reduced_cpcfg'
        return p

    def syn(p):
        p.encoder = 'syn_gcn'
        p.decoder_init_strategy = "avg_all"
        p.syn_gcn_activation = 'mish'
        p.num_enc_layers = 2
        p.enc_out_dim = 200
        return p

    encoders: Dict[str, MODIFIER] = {}

    def _add_encoder(func):
        encoders[func.__name__] = func

    _add_encoder(seq)
    _add_encoder(onlstm)
    _add_encoder(diora)
    _add_encoder(pnp)
    _add_encoder(tdpcfg)
    _add_encoder(rtdpcfg)
    _add_encoder(cpcfg)
    _add_encoder(rcpcfg)
    _add_encoder(syn)

    def bert(p):
        p.src_namespace = None
        p.decoder_init_strategy = "avg_all"
        p.lr_scheduler_kwargs = None
        p.plm_name = "bert-base-uncased"
        p.encoder = f'plm:{p.plm_name}'
        return p

    def electra(p):
        p = bert(p)
        p.plm_name = "google/electra-base-discriminator"
        p.encoder = f'plm:{p.plm_name}'
        return p

    _add_encoder(bert)
    _add_encoder(electra)
    return encoders


def decoder_modifiers() -> MOD_DICT:
    def identity(p): return p

    def onlstm(p):
        p.decoder = 'onlstm'
        return p

    decoders = {
        'seq': identity,
        'onlstm': onlstm,
        'prod': identity,
    }
    return decoders


def decorate_with(*mods):
    def decorator(f):
        @wraps(f)
        def wrapper():
            p = f()
            for m in mods:
                p = m(p)
            return p
        return wrapper

    return decorator


def install_hparamsets(base_func):
    from trialbot.training import Registry
    emods = encoder_modifiers()
    dmods = decoder_modifiers()

    from itertools import product
    for (e, emod), (d, dmod) in product(emods.items(), dmods.items()):
        hp_name = f'{e}2{d}'
        Registry._hparamsets[hp_name] = decorate_with(emod, dmod)(base_func)


def install_runtime_modifiers(hp_name: str, mod_or_mods: Union[Collection[MODIFIER], MODIFIER]):
    from trialbot.training import Registry
    from collections.abc import Collection
    if not mod_or_mods:
        return

    mods = mod_or_mods if isinstance(mod_or_mods, Collection) else [mod_or_mods]
    if len(mods) == 0:
        return

    hp_func = Registry._hparamsets[hp_name]
    Registry._hparamsets[hp_name] = decorate_with(*mods)(hp_func)


def param_overwriting_modifier(p, **kwargs):
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p
