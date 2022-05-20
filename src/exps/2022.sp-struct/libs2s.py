import argparse
from trialbot.training import TrialBot, Events


def get_updater(bot: TrialBot):
    args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
    from utils.select_optim import select_optim

    params = model.parameters()
    optim = select_optim(p, params)
    logger.info(f"Using Optimizer {optim}")

    from trialbot.data.iterators import RandomIterator
    iterator = RandomIterator(len(bot.train_set), p.batch_sz)
    logger.info(f"Using RandomIterator: on volume={len(bot.train_set)} with batch={p.batch_sz}")

    lr_scheduler_kwargs = getattr(p, 'lr_scheduler_kwargs', None)
    if lr_scheduler_kwargs is not None:
        from allennlp.training.learning_rate_schedulers import NoamLR
        lr_scheduler = NoamLR(optimizer=optim, **lr_scheduler_kwargs)
        bot.scheduler = lr_scheduler
        logger.info(f'the scheduler enabled: {lr_scheduler}')

        def _sched_step(bot: TrialBot):
            bot.scheduler.step_batch()
            if bot.state.iteration % 100 == 0:
                logger.info(f"update lr to {bot.scheduler.get_values()}")

        bot.add_event_handler(Events.ITERATION_COMPLETED, _sched_step, 100)

    from trialbot.training.updaters.training_updater import TrainingUpdater
    updater = TrainingUpdater(bot.train_set, bot.translator, model, iterator, optim, args.device, args.dry_run)
    return updater


def setup_common_bot(args: argparse.Namespace, get_model_func=None):
    assert args is not None
    from models.base_s2s.model_factory import Seq2SeqBuilder
    get_model_func = get_model_func or Seq2SeqBuilder.from_param_and_vocab
    bot = TrialBot(trial_name='enc2dec', get_model_func=get_model_func, args=args, clean_engine=True)

    from utils.trialbot.extensions import print_hyperparameters
    from utils.trialbot.extensions import get_metrics, print_models
    from trialbot.training.extensions import ext_write_info, current_epoch_logger, time_logger, loss_reporter

    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)
    bot.add_event_handler(Events.STARTED, print_models, 100)
    bot.add_event_handler(Events.STARTED, ext_write_info, 100, msg="TrailBot started")
    bot.add_event_handler(Events.STARTED, time_logger, 99)
    bot.add_event_handler(Events.STARTED, ext_write_info, 105, msg=("====" * 20))

    bot.add_event_handler(Events.EPOCH_STARTED, ext_write_info, 105, msg=("----" * 20))
    bot.add_event_handler(Events.EPOCH_STARTED, ext_write_info, 100, msg="Epoch started")
    bot.add_event_handler(Events.EPOCH_STARTED, current_epoch_logger, 99)
    bot.add_event_handler(Events.ITERATION_COMPLETED, loss_reporter, 100)

    bot.add_event_handler(Events.COMPLETED, time_logger, 101)
    bot.add_event_handler(Events.COMPLETED, ext_write_info, 100, msg="TrailBot completed.")

    if not args.test:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trialbot.extensions import end_with_nan_loss, reset_variational_dropout
        from utils.trialbot.extensions import evaluation_on_dev_every_epoch, collect_garbage
        bot.add_event_handler(Events.ITERATION_STARTED, collect_garbage, 100)
        bot.add_event_handler(Events.ITERATION_STARTED, reset_variational_dropout, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 80, on_test_data=True)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100, prefix="Training Metrics: ")
        bot.updater = get_updater(bot)
    else:
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100, prefix="Testing Metrics")

    return bot


def base_hparams():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 150
    p.WEIGHT_DECAY = 0.
    p.OPTIM = "adabelief"
    p.ADAM_BETAS = (0.9, 0.999)
    p.batch_sz = 16

    p.lr_scheduler_kwargs = {'model_size': 400, 'warmup_steps': 50}
    p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"

    p.hidden_sz = 300
    p.dropout = .5
    p.decoder = "lstm"
    p.max_decoding_step = 100
    p.scheduled_sampling = .1

    p.num_enc_layers = 1
    p.num_dec_layers = 1

    p.tied_decoder_embedding = False
    p.emb_sz = 100

    p.enc_out_dim = p.hidden_sz
    p.dec_in_dim = p.hidden_sz
    p.dec_out_dim = p.hidden_sz

    p.enc_dec_trans_usage = 'consistent'
    p.enc_dec_trans_act = 'mish'
    p.enc_dec_trans_forced = True

    p.proj_in_dim = p.emb_sz

    p.enc_dropout = 0
    p.dec_dropout = 0.5
    p.enc_attn = "dot_product"
    p.dec_hist_attn = "none"
    p.dec_inp_composer = 'cat_mapping'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'cat_mapping'
    p.proj_inp_comp_activation = 'mish'

    p.decoder_init_strategy = "forward_last_all"
    p.encoder = 'bilstm'
    p.use_cell_based_encoder = False
    # cell-based encoders: typed_rnn, ind_rnn, onlstm, lstm, gru, rnn; see models.base_s2s.base_seq2seq.py file
    # seq-based encoders: lstm, transformer, bilstm, aug_lstm, aug_bilstm; see models.base_s2s.stacked_encoder.py file
    p.cell_encoder_is_bidirectional = True     # any cell-based RNN encoder above could be bidirectional
    p.cell_encoder_uses_packed_sequence = False

    return p


if __name__ == '__main__':
    from trialbot.training import Registry
    import sys
    import os.path as osp
    sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))  # up to src
    # import datasets.cfq
    # import datasets.cfq_translator

    from utils.trialbot.setup_cli import setup
    import datasets.comp_gen_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    import datasets.cg_bundle_translator

    setup_common_bot(setup(seed=2021)).run()
