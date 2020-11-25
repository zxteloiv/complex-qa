import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
import json

from trialbot.training import TrialBot
from trialbot.training import Registry
from utils.root_finder import find_root
from utils.trialbot_setup import setup
from trialbot.utils.move_to_device import move_to_device
from build_model import lm_ebnf

import datasets.cfq
import datasets.cfq_translator

@Registry.hparamset()
def cfq_pattern():
    from trialbot.training.hparamset import HyperParamSet
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 1000
    p.emb_sz = 64
    p.hidden_dim = 64
    p.num_expander_layer = 2
    p.dropout = 0.2
    p.batch_sz = 32
    p.stack_capacity = 100

    # When using a tied embedding, the projection-based quant is conflicted with max_norm embedding,
    # which is an inplace operation and modifies the graph.
    p.tied_nonterminal_emb = True
    p.tied_terminal_emb = True
    p.nt_emb_max_norm = None   # Optional[int], set to None rather than 0 to disable max_norm
    p.t_emb_max_norm = None
    p.nt_pred_crit = "projection" # projection based or distance based
    p.t_pred_crit = "projection"

    p.grammar_entry = 'queryunit'
    p.weight_decay = 0.2

    return p

from trialbot.training.updater import TrainingUpdater, TestingUpdater
class GrammarTrainingUpdater(TrainingUpdater):
    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'GrammarTrainingUpdater':
        updater = super().from_bot(bot)
        del updater._optims
        args, hparams, model = bot.args, bot.hparams, bot.model
        from radam import RAdam
        optim = RAdam(model.parameters(), weight_decay=hparams.weight_decay)
        bot.logger.info("Use RAdam optimizer: " + str(optim))
        updater._optims = [optim]
        return updater

class GrammarTestingUpdater(TestingUpdater):
    def update_epoch(self):
        model, iterator, device = self._models[0], self._iterators[0], self._device
        model.eval()
        batch = next(iterator)
        if iterator.is_new_epoch:
            self.stop_epoch()
        if device >= 0:
            batch = move_to_device(batch, device)
        output = model(**batch)
        return output

def main():
    args = setup(seed=2021)
    from trialbot.training import Events
    bot = TrialBot(args, 'ebnf_lm', lm_ebnf)

    def get_metrics(bot: TrialBot):
        print(json.dumps(bot.model.get_metric(reset=True)))

    from utils.trial_bot_extensions import print_hyperparameters
    from trialbot.training.extensions import ext_write_info
    from trialbot.training.extensions import every_epoch_model_saver
    from utils.trial_bot_extensions import debug_models, end_with_nan_loss

    bot.add_event_handler(Events.STARTED, print_hyperparameters, 100)
    bot.add_event_handler(Events.STARTED, ext_write_info, 105, msg="-" * 50)
    bot.add_event_handler(Events.STARTED, debug_models, 100)
    bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 90)
    if not args.test:
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.updater = GrammarTrainingUpdater.from_bot(bot)
    else:
        bot.updater = GrammarTestingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()