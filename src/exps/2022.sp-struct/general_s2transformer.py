from trialbot.training import Registry
from general_s2s import encoder_decorators, guess_translator, decoder_decorators


def main():
    from utils.trialbot.setup_cli import setup as setup_cli
    from libs2s import setup_common_bot
    from models.transformer.model_factory import TransformerBuilder

    import datasets.comp_gen_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    import datasets.cg_bundle_translator

    install_hparamsets()

    args = setup_cli(seed=2021, device=0)
    if not hasattr(args, 'translator') or not args.translator:
        args.translator = guess_translator(args.hparamset)
    bot = setup_common_bot(args=args, get_model_func=TransformerBuilder.from_param_and_vocab,
                           trialname=f'trans-{args.hparamset}')
    bot.run()


def install_hparamsets():
    def _compose_func(funcname, efunc, dfunc):
        from models.transformer.model_factory import TransformerBuilder

        def _func():
            return dfunc(efunc(TransformerBuilder.base_hparams()))

        setattr(_func, '__name__', funcname)
        setattr(_func, '__qualname__', funcname)
        return _func

    encoders = encoder_decorators()
    decoders = decoder_decorators()
    for ename, efunc in encoders.items():
        for dname, dfunc in decoders.items():
            hp_name = f'{ename}2{dname}'
            Registry._hparamsets[hp_name] = _compose_func(hp_name, efunc, dfunc)


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    main()
