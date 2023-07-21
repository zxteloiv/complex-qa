import argparse
from trialbot.training import TrialBot, Events


# an alias to the mappings


def get_updater(bot: TrialBot):
    args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
    from utils.select_optim import select_optim

    if isinstance(getattr(p, 'encoder', None), str) and p.encoder.startswith('plm:'):
        bert_params, nonbert_params = [], []
        for name, param in model.named_parameters():
            if 'pretrained_model' in name:  # the attribute is used by SeqPLMEmbedEncoder wrapper.
                bert_params.append(param)
            else:
                nonbert_params.append(param)

        params = [{'lr': 1e-5, 'params': bert_params}, {'params': nonbert_params}]
    else:
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
    updater = TrainingUpdater(bot.train_set, bot.translator, model, iterator, optim, args.device, p.GRAD_CLIPPING)
    return updater


# program specific bot configurations, such as
# - experiment names
# - extensions
# - updater (optimizers)
def setup_common_bot(args: argparse.Namespace, get_model_func=None, trialname: str = None):
    assert args is not None
    from models.base_s2s.model_factory import Seq2SeqBuilder
    get_model_func = get_model_func or Seq2SeqBuilder.from_param_and_vocab
    trialname = trialname or f'enc2dec-{args.hparamset}'
    bot = TrialBot(trial_name=trialname, get_model_func=get_model_func, args=args)
    from utils.trialbot.setup_bot import setup_bot, add_metric_printing

    if not args.test:
        bot = setup_bot(bot, epoch_model_saving=False)
        from trialbot.training.extensions import every_epoch_model_saver
        if args.seed == 2021 or args.seed == '2021':
            bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.updater = get_updater(bot)
    else:
        bot = add_metric_printing(bot, 'Testing Metrics')

    return bot


if __name__ == '__main__':
    from trialbot.training import Registry
    import sys
    import os.path as osp
    sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '../../exps', '..')))  # up to src
    # import datasets.cfq
    # import datasets.cfq_translator

    from utils.trialbot.setup_cli import setup
    import shujuji.cg_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)

    setup_common_bot(setup(seed=2021)).run()
