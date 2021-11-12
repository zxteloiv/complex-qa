from trialbot.training import TrialBot, State


def reset(bot: TrialBot, *, reset_iterator: bool = True, reset_optimizer: bool = False):
    bot.state = State(epoch=0, iteration=0, output=None)
    updater = bot.updater
    if updater is None:
        return

    if reset_iterator:
        for iterator in updater._iterators:
            if hasattr(iterator, 'reset'):
                iterator.reset()

    if reset_optimizer:
        for opt in updater._optims:
            if hasattr(opt, 'reset'):
                opt.reset()
