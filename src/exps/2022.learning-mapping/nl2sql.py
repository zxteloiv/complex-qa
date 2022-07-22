import sys

from trialbot.training import TrialBot, Registry
from trialbot.utils.root_finder import find_root
sys.path.insert(0, find_root('.SRC'))


def main():
    from datasets.squall import install_squall_datasets
    from utils.trialbot.setup_cli import setup as setup_cli
    import datasets.squall_translator
    install_squall_datasets()

    args = setup_cli(seed=2021, translator='squall-base', dataset='squall0', hparamset='squall-base')
    bot = setup_bot(args)
    bot.run()


@Registry.hparamset('squall-base')
def base_param():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 80
    p.WEIGHT_DECAY = 0
    p.OPTIM = "adabelief"
    p.optim_kwargs = {"rectify": False}
    p.ADAM_LR = 1e-3
    p.ADAM_BETAS = (0.9, 0.999)
    p.batch_sz = 32
    p.emb_sz = 256
    p.hidden_sz = 256
    p.plm_model = 'bert-base-uncased'

    p.ns_keyword = 'keyword'
    p.ns_coltype = 'col_type'
    p.TRANSLATOR_KWARGS = {'ns_keyword': p.ns_keyword, 'ns_coltype': p.ns_coltype}

    p.dropout = 0.2

    p.decoder = 'lstm'
    p.num_dec_layers = 2

    p.num_heads = 1     # heads for attention only
    p.word_col_attn = 'generalized_bilinear'    # must be matrix attention
    p.col_word_attn = 'generalized_bilinear'
    p.sql_word_attn = 'mha'
    p.sql_col_attn = 'generalized_bilinear'
    p.plm_encoder = 'aug_bilstm'
    p.plm_enc_out = p.hidden_sz // 2  # = hid_sz or hid_sz//2 when encoder is bidirectional
    p.plm_enc_layers = 1

    p.decoder_init = 'zero_all'

    p.col_copy = 'generalized_bilinear'
    p.span_begin = 'generalized_bilinear'
    p.span_end = 'generalized_bilinear'
    p.bilinear_use_linear = False
    p.bilinear_use_bias = False

    # default
    p.supervise_unaligned_attn_mat = True
    p.supervise_unaligned_attn_vec = True
    return p


@Registry.hparamset()
def p_tuning_2():
    p = base_param()
    p.prompt_length = 2
    return p


@Registry.hparamset()
def p_tuning_8():
    p = base_param()
    p.prompt_length = 8
    return p


@Registry.hparamset()
def p_tuning_16():
    p = base_param()
    p.prompt_length = 16
    return p


@Registry.hparamset()
def p_tuning_32():
    p = base_param()
    p.prompt_length = 32
    return p


@Registry.hparamset()
def p_tuning_64():
    p = base_param()
    p.prompt_length = 64
    return p


@Registry.hparamset()
def p_tuning_128():
    p = base_param()
    p.prompt_length = 128
    return p


@Registry.hparamset()
def p_tuning_256():
    p = base_param()
    p.prompt_length = 256
    return p


def setup_bot(args, get_model_func=None, trialname='base'):
    from trialbot.training import Events
    from models.nl2sql.squall_base_factory import SquallBaseBuilder
    get_model_func = get_model_func or SquallBaseBuilder.from_param_vocab
    bot = TrialBot(trial_name=trialname, get_model_func=get_model_func, args=args, clean_engine=True)

    from utils.trialbot.extensions import print_hyperparameters
    from utils.trialbot.extensions import get_metrics, print_models
    from trialbot.training.extensions import ext_write_info, loss_reporter, time_logger, current_epoch_logger

    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)
    bot.add_event_handler(Events.STARTED, print_models, 100)
    bot.add_event_handler(Events.STARTED, ext_write_info, 105, msg=("====" * 20))
    bot.add_event_handler(Events.EPOCH_STARTED, ext_write_info, 105, msg=("----" * 20))
    bot.add_event_handler(Events.EPOCH_STARTED, ext_write_info, 100, msg="Epoch started")
    bot.add_event_handler(Events.EPOCH_STARTED, current_epoch_logger, 99)
    bot.add_event_handler(Events.STARTED, ext_write_info, 100, msg="TrailBot started")
    bot.add_event_handler(Events.STARTED, time_logger, 99)
    bot.add_event_handler(Events.COMPLETED, time_logger, 101)
    bot.add_event_handler(Events.COMPLETED, ext_write_info, 100, msg="TrailBot completed.")
    bot.add_event_handler(Events.ITERATION_COMPLETED, loss_reporter, 100)

    if not args.test:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trialbot.extensions import end_with_nan_loss
        from utils.trialbot.extensions import evaluation_on_dev_every_epoch, collect_garbage
        from utils.trialbot.extensions import reset_variational_dropout
        bot.add_event_handler(Events.ITERATION_STARTED, reset_variational_dropout, 100)
        bot.add_event_handler(Events.ITERATION_STARTED, collect_garbage, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)
        # bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 80, on_test_data=True)
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100, prefix="Training Metrics: ")
        bot.updater = get_updater(bot)
    else:
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100, prefix="Testing Metrics")

    return bot


def get_updater(bot: TrialBot):
    args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
    from utils.select_optim import select_optim

    bert_params, nonbert_params = [], []
    for name, param in model.named_parameters():
        if 'pretrained_model' in name:  # the attribute is used by SeqPLMEmbedEncoder wrapper.
            bert_params.append(param)
        else:
            nonbert_params.append(param)

    if getattr(p, 'prompt_length', 0) > 0:
        params = nonbert_params
    else:
        params = [{'lr': 1e-5, 'params': bert_params}, {'params': nonbert_params}]

    optim = select_optim(p, params)
    logger.info(f"Using Optimizer {optim}")

    from trialbot.data.iterators import RandomIterator
    iterator = RandomIterator(len(bot.train_set), p.batch_sz)
    logger.info(f"Using RandomIterator: on volume={len(bot.train_set)} with batch={p.batch_sz}")

    from trialbot.training.updaters.training_updater import TrainingUpdater
    updater = TrainingUpdater(bot.train_set, bot.translator, model, iterator, optim, args.device, args.dry_run)
    return updater


if __name__ == '__main__':
    main()
