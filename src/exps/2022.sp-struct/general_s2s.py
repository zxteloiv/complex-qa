import sys, logging
from trialbot.utils.root_finder import find_root
sys.path.insert(0, find_root('.SRC'))

from typing import List

from trialbot.training import Registry
from functools import wraps
from utils.s2s_arch.hparam_modifiers import (
    MODIFIER, MOD_DICT, decorate_with, install_hparamsets, install_runtime_modifiers,
)
from utils.s2s_arch.translators import (
    translator_kwargs_pool, get_translator_kwargs,
    TRANSLATOR_KWARGS_LOOKUP, install_general_translators,
    guess_translator
)


def main():
    from utils.trialbot.setup_cli import setup as setup_cli
    from utils.s2s_arch.setup_bot import setup_common_bot
    from utils.s2s_arch.base_hparams import base_hparams
    from shujuji import cogs
    import shujuji.cg_bundle as cg_bundle
    import shujuji.smcalflow_cs as smcalflow_cs

    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    cogs.install_dataset()
    smcalflow_cs.install()

    install_general_translators()
    install_hparamsets(base_hparams)

    args = setup_cli(seed=2021, device=-1)
    if not hasattr(args, 'translator') or not args.translator:
        args.translator = guess_translator(*args.hparamset.split('2'))

    install_runtime_modifiers(args, get_runtime_modifiers)

    bot = setup_common_bot(args=args)
    bot.run()


def get_runtime_modifiers(args) -> List[MODIFIER]:
    ds_mods = dataset_modifiers()
    t_mods = translator_modifiers()
    logging.debug(f"dataset-mods: {list(ds_mods.keys())} {list(ds_mods.values())}")
    logging.debug(f"translator-mods: {list(t_mods.keys())} {list(t_mods.values())}")

    selections = []
    # database modifiers, only relied on the dataset arg
    for ds, ds_mod in ds_mods.items():
        if args.dataset.startswith(ds):
            selections.append(ds_mod)
            logging.debug(f'selected----->: {ds}:{ds_mod}')

    # translator modifiers, depend on the (dataset, hparam) pair
    for tname, t_mod in t_mods.items():
        ds_prefix, general_t = tname.split('_')
        if args.dataset.startswith(ds_prefix) and args.translator == general_t:
            selections.append(t_mod)
            logging.debug(f'selected----->: {tname}:{t_mod}')

    return selections


# datasets we consider include the followings
# later we will make the references across this source file consistent.
# cogs, atis, geo, sch, adv, cfq, the names are considered as prefixes.
def dataset_modifiers() -> MOD_DICT:
    def cogs(p):
        p.TRAINING_LIMIT = 15
        p.batch_sz = 16
        p.TRANSLATOR_KWARGS = dict()
        return p

    def tiny(p):
        p.TRAINING_LIMIT = 400
        p.batch_sz = 16
        p.TRANSLATOR_KWARGS = dict()
        return p

    def medium(p):
        p.TRAINING_LIMIT = 40
        p.batch_sz = 16
        p.TRANSLATOR_KWARGS = dict()
        return p

    return {
        'cogs': cogs,
        'geo': tiny,
        'sch': tiny,
        'ati': medium,
        'adv': medium,
        # 'cfq': cogs,
        'smc': cogs,
    }


def translator_modifiers() -> MOD_DICT:
    def get_cogs_pool(p):
        # default plm_name to None, s.t. it will raise an error when it's required but absent
        return translator_kwargs_pool('nl', 'lf', source_max_token=50, use_lower_case=False,
                                      auto_plm_name=getattr(p, 'plm_name', None))

    def get_cg_pool(p):
        return translator_kwargs_pool('sent', 'sql', source_max_token=20, target_max_token=200,
                                      auto_plm_name=getattr(p, 'plm_name', None))

    def get_smc_pool(p):
        return translator_kwargs_pool('utterance', 'plan', source_max_token=40, target_max_token=200,
                                      auto_plm_name=getattr(p, 'plm_name', None))

    default_pool_factory_by_ds = {
        'cogs': get_cogs_pool,
        'geo': get_cg_pool,
        'sch': get_cg_pool,
        'ati': get_cg_pool,
        'adv': get_cg_pool,
        # 'cfq': cogs,
        'smc': get_smc_pool,
    }

    # modifiers can not be create in a loop context, nor as lambda functions,
    # and thus create them in a function context
    def create_modifier(pool_factory, translator):
        # default modifiers inherit parameters directly from the param pool
        def modifier(p):
            pool = pool_factory(p)
            p.TRANSLATOR = translator
            p.TRANSLATOR_KWARGS = get_translator_kwargs(pool, translator)
            return p
        return modifier

    general_translators = list(TRANSLATOR_KWARGS_LOOKUP.keys())
    modifiers = {}
    for ds, pool_factory in default_pool_factory_by_ds.items():
        for translator in general_translators:
            modifier_name = f'{ds}_{translator}'
            modifier = create_modifier(pool_factory, translator)
            setattr(modifier, '__name__', modifier_name)
            setattr(modifier, '__qualname__', modifier_name)
            modifiers[modifier_name] = modifier

    # some modifiers directly select and inherit hparams from the pool,
    # while others need small modifications
    # this modification can be set manually, or by rule.
    # e.g., translators for the cogs dataset ending with "prod" use different field
    def create_updated_modifiers(modifier):
        @wraps(modifier)
        def prod_rule(p):
            p = modifier(p)
            default = p.TRANSLATOR_KWARGS['target_field']
            p.TRANSLATOR_KWARGS.update(target_field=f'{default}_tree')
            return p
        return prod_rule

    for mod_name in modifiers.keys():
        if mod_name.endswith('prod'):
            modifiers[mod_name] = create_updated_modifiers(modifiers[mod_name])

    return modifiers


if __name__ == '__main__':
    main()
