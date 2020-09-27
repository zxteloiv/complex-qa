import sys
import logging
import json

from trialbot.training import TrialBot
from trialbot.training import Registry
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from utils.root_finder import find_root
from trialbot.utils.move_to_device import move_to_device
from utils.trialbot_setup import setup

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
    p.hidden_dim = 64
    p.num_layers = 2
    p.dropout = .2
    p.weight_decay = .2

    return p

@Registry.hparamset()
def cfq_lstm_one_layer():
    p = cfq_pattern()
    p.num_layers = 1
    return p

def get_model(p, vocab):
    from trialbot.data.ns_vocabulary import NSVocabulary
    vocab: NSVocabulary
    import torch.nn as nn
    from models.modules.mixture_softmax import MoSProjection
    from models.matching.seq_modeling import SeqModeling, PytorchSeq2SeqWrapper
    m = SeqModeling(
        embedding=nn.Embedding(vocab.get_vocab_size(p.target_namespace), p.token_dim),
        encoder=PytorchSeq2SeqWrapper(nn.LSTM(p.token_dim, p.hidden_dim, p.num_layers, batch_first=True,
                                              dropout=p.dropout if p.num_layers > 1 else 0.)),
        padding=0,
        prediction=MoSProjection(5, p.token_dim, vocab.get_vocab_size(p.target_namespace)),
        attention=None,
    )
    return m

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
    args = setup(seed="2021", hparamset="cfq_pattern", dataset="cfq_mcd1", translator="cfq")
    bot = TrialBot(trial_name="lstm_lm", get_model_func=get_model, args=args)

    from trialbot.training import Events
    def training_metrics(bot: TrialBot):
        print(json.dumps(bot.model.get_metric(reset=True)))

    if args.test:
        from trialbot.training.trial_bot import Engine
        new_engine = Engine()
        new_engine.register_events(*Events)
        bot._engine = new_engine
        bot.add_event_handler(Events.EPOCH_COMPLETED, training_metrics, 90)

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def print_output(bot: TrialBot):
            output = bot.state.output
            if output is None:
                return

            for l, raw in zip(output['likelihoods'], output['_raw']):
                to_print = {
                    "likelihood": l.item(),
                    "reconstructed_sparqlPattern": raw['reconstructed_sparql_pattern'],
                    "sparqlPattern": raw['example']['sparqlPattern'],
                }
                print(json.dumps(to_print))

        bot.updater = CFQTestingUpdater.from_bot(bot)
    else:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import debug_models, end_with_nan_loss

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, training_metrics, 90)
        bot.add_event_handler(Events.STARTED, debug_models, 100)
        bot.updater = CFQTrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()
