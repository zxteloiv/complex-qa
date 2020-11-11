import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
import logging
import json

from trialbot.training import TrialBot
from trialbot.training import Registry
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from utils.root_finder import find_root
from trialbot.utils.move_to_device import move_to_device
from utils.trialbot_setup import setup
from .build_model import lm_npda

import datasets.cfq
import datasets.cfq_translator

ROOT = find_root()

@Registry.hparamset()
def cfq_pattern():
    from trialbot.training.hparamset import HyperParamSet
    p = HyperParamSet.common_settings(ROOT)
    p.TRAINING_LIMIT = 50
    p.batch_sz = 128
    p.target_namespace = 'sparqlPattern'
    p.token_dim = 64
    p.stack_dim = 64
    p.hidden_dim = 64
    p.ntdec_layer = 1
    p.dropout = .2
    p.stack_capacity = 150
    p.ntdec_factor = 1.
    p.weight_decay = .2
    p.pda_type = 'trnn'
    p.ntdec_normalize = True

    # next_state, current_token (t4nt),
    # last_state_default_none, last_state_default_current
    p.ntdec_init = "last_state_default_none"

    p.codebook = "split"  # vanilla, split
    # stack_dim must be divisible by split_num (N)
    p.split_num = 4
    # num_nonterminals had better be set such that the N-root of it is an integer.
    # otherwise the compositional num_nt will not be as specified.
    # actually compositional_num_nt = round(num_nt ** (1/N)) ^ N
    p.num_nonterminals = 16
    p.codebook_decay = 0.99
    p.codebook_initial_n = 1

    return p

from utils.trialbot_grid_search_helper import import_grid_search_parameters
import_grid_search_parameters(
    grid_conf={
        "ntdec_normalize": [True, False],
        "num_nonterminals": [20, 30, 50],
    },
    base_param_fn=cfq_pattern,
    name_prefix="ntnorm_ntnum_",
)

class CFQTrainingUpdater(TrainingUpdater):
    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()
        batch = next(iterator)

        sp = batch['sparqlPattern']
        if device >= 0:
            sp = move_to_device(sp, device)

        output = model(seq=sp)

        if not self._dry_run:
            loss = output['loss']
            loss.backward()
            optim.step()
            model.npda.codebook.update_codebook()
        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'CFQTrainingUpdater':
        updater = super().from_bot(bot)
        del updater._optims
        args, hparams, model = bot.args, bot.hparams, bot.model
        from radam import RAdam
        optim = RAdam(model.parameters(), weight_decay=hparams.weight_decay)
        bot.logger.info("Use RAdam optimizer: " + str(optim))
        updater._optims = [optim]
        return updater

class CFQTestingUpdater(TestingUpdater):
    def update_epoch(self):
        model, iterator = self._models[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.eval()
        batch = next(iterator)
        sp = batch['sparqlPattern']
        if device >= 0:
            sp = move_to_device(sp, device)
        output = model(seq=sp)
        output['_raw'] = batch['_raw']
        return output

def main():
    args = setup(seed="2021", dataset="cfq_mcd1", translator="cfq")
    bot = TrialBot(trial_name="npda_lm", get_model_func=lm_npda, args=args)

    from trialbot.training import Events
    def get_metrics(bot: TrialBot):
        print(json.dumps(bot.model.get_metric(reset=True)))

    if args.test:
        from trialbot.training.trial_bot import Engine
        new_engine = Engine()
        new_engine.register_events(*Events)
        bot._engine = new_engine
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 90)

        from utils.trial_bot_extensions import print_hyperparameters
        from trialbot.training.extensions import ext_write_info
        bot.add_event_handler(Events.STARTED, print_hyperparameters, 100)
        bot.add_event_handler(Events.STARTED, ext_write_info, 105, msg="-" * 50)

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def print_output(bot: TrialBot):
            output = bot.state.output
            if output is None:
                return

            debug = bot.args.debug
            zip_resource = [output['likelihoods'], output['_raw']]
            if debug:
                zip_resource.append(output['batch_replays'])

            for data_tuple in zip(*zip_resource):
                l, raw = data_tuple[:2]
                to_print = {
                    "likelihood": l.item(),
                    "reconstructed_sparqlPattern": raw['reconstructed_sparql_pattern'],
                    "sparqlPattern": raw['example']['sparqlPattern'],
                }
                if debug:
                    replays = data_tuple[-1]
                    to_print['replays'] = replays
                print(json.dumps(to_print))

        bot.updater = CFQTestingUpdater.from_bot(bot)
    else:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import debug_models, end_with_nan_loss

        from utils.trial_bot_extensions import print_hyperparameters
        from utils.trial_bot_extensions import track_pytorch_module_forward_time
        from trialbot.training.extensions import ext_write_info
        bot.add_event_handler(Events.STARTED, print_hyperparameters, 100)
        bot.add_event_handler(Events.STARTED, ext_write_info, 105, msg="-" * 50)
        bot.add_event_handler(Events.STARTED, track_pytorch_module_forward_time, 105, max_depth=3)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 90)
        bot.add_event_handler(Events.STARTED, debug_models, 100)
        bot.updater = CFQTrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()
