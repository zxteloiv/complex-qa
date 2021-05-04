from typing import Dict, List, Tuple, Callable, Any, Union
from os.path import join, abspath, dirname
import sys
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..')))   # up to src
import torch
from trialbot.training import TrialBot, Events, Registry, Updater
from trialbot.data.iterators import RandomIterator
from trialbot.utils.move_to_device import move_to_device

import logging
import datasets.cfq
import datasets.cfq_translator
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_geo_qa_datasets()
from utils.trialbot_setup import install_dataset_into_registry
install_dataset_into_registry(cg_bundle.CG_DATA_REG)

import datasets.cg_bundle_translator

def get_tranx_updater(bot: TrialBot):
    args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
    from utils.select_optim import select_optim
    from utils.maybe_random_iterator import MaybeRandomIterator

    params = model.parameters()
    optim = select_optim(p, params)
    logger.info(f"Using Optimizer {optim}")

    device, dry_run = args.device, args.dry_run
    repeat_iter = shuffle_iter = not args.debug
    iterator = MaybeRandomIterator(bot.train_set, p.batch_sz, bot.translator, shuffle=shuffle_iter, repeat=repeat_iter)
    if args.debug and args.skip:
        iterator.reset(args.skip)

    from trialbot.training.updater import TrainingUpdater
    updater = TrainingUpdater(model, iterator, optim, device, dry_run)
    return updater

def main():
    from utils.trialbot_setup import setup
    from models.base_s2s.base_seq2seq import BaseSeq2Seq
    args = setup(seed=2021, translator='cfq_tranx_mod_ent_qa', dataset='cfq_mcd1', hparamset='cfq_mod_ent_tranx')
    bot = TrialBot(trial_name='tranx_qa', get_model_func=BaseSeq2Seq.from_param_and_vocab, args=args)

    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def get_metrics(bot: TrialBot):
        import json
        print(json.dumps(bot.model.get_metric(reset=True)))

    from utils.trial_bot_extensions import print_hyperparameters
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)

    from trialbot.training import Events
    if not args.test:
        # --------------------- Training -------------------------------
        @bot.attach_extension(Events.STARTED)
        def print_models(bot: TrialBot):
            print(str(bot.models))

        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import end_with_nan_loss
        from utils.trial_bot_extensions import evaluation_on_dev_every_epoch, collect_garbage
        bot.add_event_handler(Events.EPOCH_STARTED, collect_garbage, 95)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)

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
            for src, gold, pred in zip(*map(output.get, ("source_tokens", "target_tokens", "predicted_tokens"))):
                to_print = f'SRC:  {" ".join(src)}\nGOLD: {" ".join(gold)}\nPRED: {" ".join(pred)}'
                batch_print.append(to_print)

            sep = '\n' + '-' * 60 + '\n'
            print(sep.join(batch_print))

    bot.run()

@Registry.hparamset()
def cfq_mod_ent_tranx():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 10  # in num of epochs
    p.OPTIM = "RAdam"
    p.batch_sz = 32

    p.emb_sz = 256
    p.src_namespace = 'questionPatternModEntities'
    p.tgt_namespace = 'modent_rule_seq'
    p.hidden_sz = 256
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "dot_product"
    p.concat_attn_to_dec_input = False
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.dropout = .2
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    p.tied_decoder_embedding = True
    return p

@Registry.hparamset()
def cfq_mod_ent_moderate_tranx():
    p = cfq_mod_ent_tranx()
    p.TRAINING_LIMIT = 10  # in num of epochs
    p.WEIGHT_DECAY = .1
    p.emb_sz = 128
    p.hidden_sz = 128
    p.dec_hist_attn = "dot_product"
    return p

@Registry.hparamset()
def cfq_mod_ent_moderate_tranx_scaled():
    p = cfq_mod_ent_tranx()
    p.TRAINING_LIMIT = 10  # in num of epochs
    p.OPTIM = "adabelief"
    p.WEIGHT_DECAY = .1
    p.ADAM_BETAS = (0.9, 0.98)
    p.optim_kwargs = {"eps": 1e-16}
    p.emb_sz = 128
    p.hidden_sz = 128
    p.dec_hist_attn = "dot_product"

    p.training_average = 'batch'
    return p

@Registry.hparamset()
def common_sql_tranx():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 100  # in num of epochs
    p.OPTIM = "adabelief"
    p.WEIGHT_DECAY = .1
    p.ADAM_BETAS = (0.9, 0.999)
    p.batch_sz = 16

    p.emb_sz = 128
    p.hidden_sz = 128
    p.src_namespace = 'sent'
    p.tgt_namespace = 'rule_seq'
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "dot_product"
    p.concat_attn_to_dec_input = False
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.dropout = .5
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    p.tied_decoder_embedding = True
    return p

from trialbot.utils.grid_search_helper import import_grid_search_parameters
import_grid_search_parameters(
    grid_conf={
        "tied_decoder_embedding": [False],
        "hidden_sz": [128, 256],
        "emb_sz": [128, 256],
    },
    base_param_fn=common_sql_tranx,
    name_prefix="hp_tuning_",
)

if __name__ == '__main__':
    main()
