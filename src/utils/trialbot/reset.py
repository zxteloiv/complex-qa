from trialbot.training import TrialBot, State


def reset(bot: TrialBot):
    bot.state = State(epoch=0, iteration=0, output=None)
    if bot.updater:
        for iterator in bot.updater._iterators:
            if hasattr(iterator, 'reset'):
                iterator.reset()
