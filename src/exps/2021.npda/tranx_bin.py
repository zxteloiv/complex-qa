import logging
import sys, os.path as osp
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))   # up to src
from trialbot.training import TrialBot, Events, Registry
from utils.trialbot.setup_cli import setup
# import datasets.cfq
# import datasets.cfq_translator
import shujuji.comp_gen_bundle as cg_bundle
cg_bundle.install_parsed_qa_datasets(Registry._datasets)
import shujuji.cg_bundle_translator
import tranx_hparamset


def get_tranx_updater(bot: TrialBot):
    args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
    from utils.select_optim import select_optim

    params = model.parameters()
    optim = select_optim(p, params)
    logger.info(f"Using Optimizer {optim}")

    cluster_id = getattr(p, 'cluster_iter_key', None)
    if cluster_id is None:
        from trialbot.data.iterators import RandomIterator
        iterator = RandomIterator(len(bot.train_set), p.batch_sz)
        logger.info(f"Using RandomIterator with batch={p.batch_sz}")
    else:
        from trialbot.data.iterators import ClusterIterator
        iterator = ClusterIterator(bot.train_set, p.batch_sz, cluster_id)
        logger.info(f"Using ClusterIterator with batch={p.batch_sz} cluster_key={cluster_id}")

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


def run_exp(args=None):
    from models.base_s2s.base_seq2seq import BaseSeq2Seq
    from trialbot.training import Events
    if args is None:
        args = setup(seed=2021)
    bot = TrialBot(trial_name='tranx_qa', get_model_func=BaseSeq2Seq.from_param_and_vocab, args=args)

    from utils.trialbot.extensions import print_hyperparameters
    from utils.trialbot.extensions import get_metrics, print_models
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)
    bot.add_event_handler(Events.STARTED, print_models, 100)

    from trialbot.training import Events
    if not args.test:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trialbot.extensions import end_with_nan_loss
        from utils.trialbot.extensions import evaluation_on_dev_every_epoch, collect_garbage
        bot.add_event_handler(Events.EPOCH_STARTED, collect_garbage, 95)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90, rewrite_eval_hparams={"batch_sz": 32})
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 80, rewrite_eval_hparams={"batch_sz": 32}, on_test_data=True)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100, prefix="Training Metrics")

        bot.updater = get_tranx_updater(bot)

    elif args.debug:
        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def print_output(bot: TrialBot):
            import json
            output = bot.state.output
            if output is None:
                return

            model = bot.model
            output = model.revert_tensor_to_string(output)

            batch_print = []
            for err, src, gold, pred in zip(*map(output.get, ("errno", "source_tokens", "target_tokens", "predicted_tokens"))):
                to_print = f'ERR:  {err}\nSRC:  {" ".join(src)}\nGOLD: {" ".join(gold)}\nPRED: {" ".join(pred)}'
                batch_print.append(to_print)

            sep = '\n' + '-' * 60 + '\n'
            print(sep.join(batch_print))
    else:
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100)

    bot.run()
    return bot


def main():
    args = dict()
    if len(sys.argv) == 1:
        args.update(dataset='scholar_iid.handcrafted', seed=2021, hparamset='scholar_common', device=0)

    run_exp(setup(translator='tranx_no_terminal', **args))


if __name__ == '__main__':
    main()
