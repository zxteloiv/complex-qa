from trialbot.training import TrialBot, Events


def setup_bot(bot: TrialBot,
              epoch_model_saving: bool = True,
              epoch_dev_eval: bool = True,
              epoch_test_eval: bool = True,
              metric_printing: bool = True,
              vdrop_reset: bool = True,
              use_gc: bool = True,
              ):
    if epoch_model_saving:
        add_model_saving(bot)
    if epoch_dev_eval:
        add_epoch_dev_eval(bot)
    if epoch_test_eval:
        add_epoch_test_eval(bot)
    if metric_printing:
        add_metric_printing(bot)
    if vdrop_reset:
        add_vdrop_reset(bot)
    if use_gc:
        add_iteration_gc(bot)
    return bot


def add_model_saving(bot: TrialBot):
    from trialbot.training.extensions import every_epoch_model_saver
    bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
    return bot


def add_epoch_dev_eval(bot: TrialBot):
    from utils.trialbot.extensions import evaluation_on_dev_every_epoch
    bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)
    return bot


def add_epoch_test_eval(bot: TrialBot):
    from utils.trialbot.extensions import evaluation_on_dev_every_epoch
    bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 80, on_test_data=True)
    return bot


def add_metric_printing(bot: TrialBot, prefix='Training Metrics: '):
    from utils.trialbot.extensions import get_metrics
    bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100, prefix=prefix)
    return bot


def add_iteration_gc(bot: TrialBot):
    from trialbot.training.extensions import collect_garbage
    bot.add_event_handler(Events.ITERATION_STARTED, collect_garbage, 100)
    return bot


def add_vdrop_reset(bot: TrialBot):
    from utils.trialbot.extensions import reset_variational_dropout
    bot.add_event_handler(Events.ITERATION_STARTED, reset_variational_dropout, 100)
    return bot


