from trialbot.training import TrialBot
from utils.tensorboard_writer import TensorboardWriter


def init_tensorboard_writer(bot: TrialBot, interval: int = 32, histogram_interval: int = 100):
    bot.tbx_writer = TensorboardWriter(bot.savepath,
                                       summary_interval=interval,
                                       histogram_interval=histogram_interval,
                                       get_iteration_fn=lambda: bot.state.iteration)


def write_batch_info_to_tensorboard(bot: TrialBot):
    output = bot.state.output
    if output is not None:
        bot.tbx_writer: TensorboardWriter
        metrics = bot.model.get_metric()
        bot.tbx_writer.log_batch(bot.model, output.get('loss', 0), metrics, output.get('param_update', None))


def close_tensorboard(bot: TrialBot):
    bot.tbx_writer: TensorboardWriter
    bot.tbx_writer.close()