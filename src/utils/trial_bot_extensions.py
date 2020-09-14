from trialbot.training import TrialBot
import os.path
import torch

def save_model_every_num_iters(bot: TrialBot, interval: int = 100):
    if bot.state.iteration % interval == 0:
        savedir, model = bot.savepath, bot.model
        filename = os.path.join(savedir, f"model_state_{bot.state.epoch}-{bot.state.iteration}.th")
        torch.save(model.state_dict(), filename)
        bot.logger.info(f"model saved to {filename}")

def save_multiple_models_every_num_iters(bot: TrialBot, interval: int = 100):
    if bot.state.iteration % interval == 0:
        savedir, models = bot.savepath, bot.models
        for model_id, model in enumerate(models):
            filename = os.path.join(savedir, f"model_{model_id}_state_{bot.state.epoch}-{bot.state.iteration}.th")
            torch.save(model.state_dict(), filename)
            bot.logger.info(f"model {model_id} saved to {filename}")

def save_multiple_models_per_epoch(bot: TrialBot, interval: int = 1):
    if bot.state.epoch % interval == 0:
        savedir, models = bot.savepath, bot.models
        for model_id, model in enumerate(models):
            filename = os.path.join(savedir, f"model_{model_id}_state_{bot.state.epoch}.th")
            torch.save(model.state_dict(), filename)
            bot.logger.info(f"model {model_id} saved to {filename}")

def debug_models(bot: TrialBot):
    bot.logger.debug(str(bot.models))

def end_with_nan_loss(bot: TrialBot):
    import numpy as np
    output = bot.state.output
    if output is None:
        return
    loss = output["loss"]
    def _isnan(x):
        if isinstance(x, torch.Tensor):
            return bool(torch.isnan(x).any())
        elif isinstance(x, np.ndarray):
            return bool(np.isnan(x).any())
        else:
            import math
            return math.isnan(x)

    if _isnan(loss):
        bot.logger.error("NaN loss encountered, training ended")
        bot.state.epoch = bot.hparams.TRAINING_LIMIT + 1
        bot.updater.stop_epoch()

def print_hyperparameters(bot: TrialBot):
    bot.logger.info(f"Hyperparamset Used: {bot.args.hparamset}")
    bot.logger.info(str(bot.hparams))
