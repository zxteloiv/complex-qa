from trialbot.training import TrialBot
import math
from datetime import datetime
from copy import copy
from trialbot.utils.move_to_device import move_to_device
import os.path
import gc
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
    output = getattr(bot.state, 'output', None)
    if output is None:
        return
    loss = output["loss"]

    def _isnan(x):
        if isinstance(x, torch.Tensor):
            return bool(torch.isnan(x).any())
        elif isinstance(x, np.ndarray):
            return bool(np.isnan(x).any())
        else:
            return math.isnan(x)

    if _isnan(loss):
        bot.logger.error("NaN loss encountered, training ended")
        bot.state.epoch = bot.hparams.TRAINING_LIMIT + 1
        bot.updater.stop_epoch()


def print_hyperparameters(bot: TrialBot):
    bot.logger.info(f"Cmd Arguments Used:\n{bot.args}")
    bot.logger.info(f"Hyperparamset Used: {bot.args.hparamset}\n{str(bot.hparams)}")


def print_models(bot: TrialBot):
    bot.logger.info("Model Specs:\n" + str(bot.models))


def track_pytorch_module_forward_time(bot: TrialBot, max_depth: int = -1, timefmt="%H:%M:%S.%f"):
    models = bot.models

    def print_time_hook(m: torch.nn.Module, inp):
        dt, micro = datetime.now().strftime(timefmt).split('.')
        time_str = "%s.%03d" % (dt, int(micro) / 1000)
        module_name = f"{m.__class__.__module__}.{m.__class__.__name__}"
        bot.logger.debug(f"module {module_name} called at {time_str}")

    for model in models:
        if len(list(model.modules())) <= 0:
            continue

        basename, _ = next(model.named_modules())
        base_depth = basename.count('.')

        for name, module in model.named_modules():
            if max_depth < 0 or name.count('.') - base_depth < max_depth:
                module.register_forward_pre_hook(print_time_hook)


def evaluation_on_dev_every_epoch(bot: TrialBot, interval: int = 1,
                                  clear_cache_each_batch: bool = True,
                                  rewrite_eval_hparams: dict = None,
                                  skip_first_epochs: int = 0,
                                  on_test_data: bool = False,
                                  ):
    if bot.state.epoch % interval == 0 and bot.state.epoch > skip_first_epochs:
        if on_test_data:
            bot.logger.info("Running for evaluation metrics on testing ...")
        else:
            bot.logger.info("Running for evaluation metrics ...")

        dataset, hparams = bot.dev_set, copy(bot.hparams)
        rewrite_eval_hparams = rewrite_eval_hparams or dict()
        for k, v in rewrite_eval_hparams.items():
            setattr(hparams, k, v)
        from trialbot.data import RandomIterator
        dataset = bot.test_set if on_test_data else bot.dev_set
        iterator = RandomIterator(len(dataset), hparams.batch_sz, shuffle=False, repeat=False)
        model = bot.model
        device = bot.args.device
        model.eval()
        for indices in iterator:
            tensor_list = [bot.translator.to_tensor(dataset[index]) for index in indices]
            batch = bot.translator.batch_tensor(tensor_list)
            if batch is None or len(batch) == 0:
                continue
            if device >= 0:
                batch = move_to_device(batch, device)
            model(**batch)

            if clear_cache_each_batch:
                import gc
                gc.collect()
                if bot.args.device >= 0:
                    import torch.cuda
                    torch.cuda.empty_cache()

        if hasattr(bot, 'tbx_writer'):
            val_metrics = bot.model.get_metric(reset=False)
            bot.tbx_writer.log_metrics(dict(), val_metrics=val_metrics)
        get_metrics(bot, prefix="Testing Metrics: " if on_test_data else "Evaluation Metrics: ")


def collect_garbage(bot: TrialBot):
    for optim in bot.updater._optims:
        optim.zero_grad()

    if hasattr(bot.state, "output") and bot.state.output is not None:
        bot.state.output = None
    gc.collect()
    if bot.args.device >= 0:
        import torch.cuda
        torch.cuda.empty_cache()


def get_metrics(bot: TrialBot, prefix: str = ""):
    import json
    for i, model in enumerate(bot.models):
        if getattr(model, 'get_metric', None):
            bot.logger.info(prefix + json.dumps(model.get_metric(reset=True)))
        elif getattr(model, 'get_metrics', None):
            bot.logger.info(prefix + json.dumps(model.get_metrics(reset=True)))
        else:
            bot.logger.warning(f'neither get_metric nor get_metrics method is found')


def print_snaptshot_path(bot: TrialBot):
    bot.logger.info("Snapshot Dir: " + bot.savepath)


def reset_variational_dropout(bot: TrialBot):
    from models.modules.variational_dropout import VariationalDropout
    for model in bot.models:
        for m in model.modules():
            if isinstance(m, VariationalDropout):
                m.reset()
