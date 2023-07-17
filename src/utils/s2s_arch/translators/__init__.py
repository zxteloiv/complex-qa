from operator import itemgetter
from typing import Dict, List, Tuple, Any, Optional
from .s2s import Seq2SeqTranslator
from .plm2s import PLM2SeqTranslator, PREPROCESS_HOOKS, AutoPLMField
from .s2prod import Seq2ProdTranslator, TerminalRuleSeqField
from .plm2prod import PLM2ProdTranslator
from .syn2s import Syn2SeqTranslator, BeNeParField
from .syn2prod import Syn2ProdTranslator


def install_general_translators(reg: dict = None):
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._translators

    reg['s2s'] = Seq2SeqTranslator
    reg['plm2s'] = PLM2SeqTranslator
    reg['s2prod'] = Seq2ProdTranslator
    reg['plm2prod'] = PLM2ProdTranslator
    reg['syn2s'] = Syn2SeqTranslator
    reg['syn2prod'] = Syn2ProdTranslator


KWARGS_POOL = list[tuple[str, Any]]

TRANSLATOR_PARAMS = (
    'source_field', # 0
    'target_field', # 1
    'source_max_token', # 2
    'target_max_token', # 3, same as default
    'use_lower_case', # 4,
    'auto_plm_name', # 5,
    'source_preprocess_hooks', # 6, same as default
    'no_preterminals', # 7, same as default
    'spacy_model', # 8, same as default
    'benepar_model', # 9, same as default
)

TRANSLATOR_KWARGS_LOOKUP: dict[str, list[int]] = {
    's2s': [0, 1, 2, 3, 4],
    'plm2s': [0, 1, 3, 4, 5, 6],
    's2prod': [0, 1, 2, 3, 4, 7],
    'plm2prod': [0, 1, 3, 4, 5, 6, 7],
    'syn2s': [0, 1, 3, 4, 6, 8, 9],
    'syn2prod': [0, 1, 3, 4, 6, 7, 8, 9],
}


def guess_translator(enc_name: str, dec_name: str) -> str:
    # return the translator name based on the hparam modifiers (encoder and decoder)
    # first part of the translator name
    syn_graph_enc = enc_name == 'syn'
    plm_enc = enc_name in ('electra', 'bert')
    if syn_graph_enc:
        t_prefix = 'syn'
    elif plm_enc:
        t_prefix = 'plm'
    else:
        t_prefix = 's'

    # second part of the translator name
    prod_dec = dec_name == 'prod'
    t_suffix = 'prod' if prod_dec else 's'

    # refer to the .translators.__init__ for the complete list of translators
    return f'{t_prefix}2{t_suffix}'


def translator_kwargs_pool(source_field: str,
                           target_field: str,
                           source_max_token: int = 0,
                           target_max_token: int = 0,
                           use_lower_case: bool = True,
                           auto_plm_name: str = 'bert-base-uncased',
                           source_preprocess_hook: PREPROCESS_HOOKS | None = None,
                           no_preterminals: bool = True,
                           spacy_model: str = 'en_core_web_md',
                           benepar_model: str = 'benepar_en3',
                           ) -> KWARGS_POOL:
    """
    Consider the Translators in the .translators only.
    The pool is a list of pairs that can be used to initialize a dict.
    Parameters of all translator constructors are aggregated here,
    and only a few of them are going to be selected.
    """
    param_vals = (
        source_field,  # 0
        target_field,  # 1
        source_max_token,  # 2
        target_max_token,  # 3, same as default
        use_lower_case,  # 4,
        auto_plm_name,  # 5,
        source_preprocess_hook,  # 6, same as default
        no_preterminals,  # 7, same as default
        spacy_model,  # 8, same as default
        benepar_model,  # 9, same as default
    )
    ordered_pool = list(zip(TRANSLATOR_PARAMS, param_vals))
    return ordered_pool


def get_translator_kwargs(pool: KWARGS_POOL, translator_name: str):
    indices: List[int] = TRANSLATOR_KWARGS_LOOKUP[translator_name]
    return dict(itemgetter(*indices)(pool))
