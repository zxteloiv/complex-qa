import itertools
import pprint

import torch
import os.path as osp
from trialbot.training import Registry
from trialbot.training import TrialBot


def main():
    from datasets.wikisql import install_dataset
    install_dataset()
    from datasets.wikisql_translator import install_translators
    install_translators()
    from utils.trialbot.setup_cli import setup as setup_cli
    from models.nl2sql.sqlova import SQLova

    args = setup_cli(seed=2021, translator='sqlova', dataset='wikisql', hparamset='sqlova')
    bot = TrialBot(trial_name='sqlova', get_model_func=SQLova.build, args=args, clean_engine=True)
    decorate_bot(args, bot)
    bot.run()


def decorate_bot(args, bot):
    from trialbot.training import Events

    from utils.trialbot.extensions import print_hyperparameters, print_snaptshot_path
    from utils.trialbot.extensions import get_metrics, print_models
    from trialbot.training.extensions import ext_write_info, loss_reporter, time_logger, current_epoch_logger

    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)
    bot.add_event_handler(Events.STARTED, print_snaptshot_path, 90)
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

    if not args.test:   # Training, not testing or evaluation
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trialbot.extensions import end_with_nan_loss
        from utils.trialbot.extensions import evaluation_on_dev_every_epoch, collect_garbage
        from utils.trialbot.extensions import reset_variational_dropout
        bot.add_event_handler(Events.ITERATION_STARTED, reset_variational_dropout, 100)
        bot.add_event_handler(Events.ITERATION_STARTED, collect_garbage, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 80, on_test_data=True)
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100, prefix="Training Metrics: ")

        def get_updater(bot: TrialBot):
            args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
            from utils.select_optim import select_optim

            bert_params, nonbert_params = [], []
            for name, param in model.named_parameters():
                if 'plm_model' in name:
                    bert_params.append(param)
                else:
                    nonbert_params.append(param)

            params = [{'lr': 1e-5, 'params': bert_params}, {'params': nonbert_params}]

            optim = select_optim(p, params)
            logger.info(f"Using Optimizer {optim}")

            from trialbot.data.iterators import RandomIterator
            iterator = RandomIterator(len(bot.train_set), p.batch_sz)
            logger.info(f"Using RandomIterator: on volume={len(bot.train_set)} with batch={p.batch_sz}")

            from trialbot.training.updaters.training_updater import TrainingUpdater
            updater = TrainingUpdater(bot.train_set, bot.translator, model, iterator, optim, args.device, args.dry_run)
            return updater

        bot.updater = get_updater(bot)

    else:
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100, prefix="Testing Metrics")

    return bot


@Registry.hparamset('sqlova')
def base_params():
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
    p.plm_model = osp.abspath(osp.expanduser('~/.cache/complex_qa/bert-base-uncased'))
    p.hidden_sz = 100
    p.num_layers = 2
    p.dropout = .3
    p.max_conds = 4
    p.TRANSLATOR_KWARGS = {'plm_name': p.plm_model}
    return p


@Registry.hparamset('sqlova-adaloss')
def adaloss():
    p = base_params()
    p.use_metric_adaptive_losses = True
    return p


@Registry.hparamset('sqlova-hungarian')
def hungarian():
    p = base_params()
    p.use_hungarian_loss = True
    return p


def debug():
    from datasets.wikisql import install_dataset
    from datasets.wikisql_translator import install_translators
    install_translators()
    install_dataset()
    train, dev, test = Registry.get_dataset('wikisql')
    plm_dir = osp.abspath(osp.expanduser('~/.cache/complex_qa/bert-base-uncased'))
    translator = Registry.get_translator('sqlova', plm_name=plm_dir)
    print(len(train), len(dev), len(test))
    for x in train[:10]:
        pprint.pprint([x[k] for k in ('question', 'sql')])
        pprint.pprint({k: v for k, v in translator.to_tensor(x).items() if not k.startswith('src_')})

    b = translator.batch_tensor([translator.to_tensor(x) for x in train[:10]])
    pprint.pprint({k: v for k, v in b.items() if not k.startswith('src_')})


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    # debug()
    main()
