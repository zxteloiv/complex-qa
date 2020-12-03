import os.path
import sys
sys.path.insert(0, os.path.join('..', '..'))
import logging
import json

from trialbot.training import TrialBot
from trialbot.training import Registry
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.utils.root_finder import find_root
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
    p.dropout = .2
    p.weight_decay = .2

    p.encoder = "lstm"  # lstm, transformer
    p.num_layers = 2
    p.predictor = "quant" # quant, mos, tied
    p.quant_criterion = "projection" # for quant predictor: distance, projection
    p.num_mixture = 5   # for mos predictor

    # by default, tied(quant is set tied)+dec_norm
    p.embedding_norm = False
    # no decoding norm set, because quant-projection predictor implies a decoding cosine norm

    return p

@Registry.hparamset()
def cfq_pattern_mod_ent():
    p = cfq_pattern()
    p.target_namespace = 'sparqlPatternModEntities'
    return p

@Registry.hparamset()
def cfq_pattern_not_norm():
    p = cfq_pattern()
    p.predictor = "tied"
    p.embedding_norm = False
    return p

@Registry.hparamset()
def cfq_pattern_emb_norm():
    p = cfq_pattern()
    p.predictor = "tied"
    p.embedding_norm = True
    return p

@Registry.hparamset()
def cfq_pattern_both_norm():
    p = cfq_pattern()
    p.predictor = "quant"
    p.quant_criterion = "projection" # for quant predictor: distance, projection
    p.embedding_norm = True
    return p

@Registry.hparamset()
def cfq_pattern_mos():
    p = cfq_pattern()
    p.predictor = "mos"
    return p

@Registry.hparamset()
def cfq_pattern_mos_n1():
    p = cfq_pattern()
    p.predictor = "mos"
    p.num_mixture = 1
    return p

@Registry.hparamset()
def cfq_pattern_quant_dist():
    p = cfq_pattern()
    p.predictor = "quant" # quant, mos
    p.quant_criterion = "distance"
    return p

def get_model(p, vocab):
    from trialbot.data.ns_vocabulary import NSVocabulary
    vocab: NSVocabulary
    import torch.nn as nn
    from models.modules.mixture_softmax import MoSProjection
    from models.matching.seq_modeling import SeqModeling, PytorchSeq2SeqWrapper
    from models.modules.quantized_token_predictor import QuantTokenPredictor
    from models.modules.normalization import Normalization

    num_toks = vocab.get_vocab_size(p.target_namespace)
    emb = nn.Embedding(num_toks, p.token_dim)
    encoder = PytorchSeq2SeqWrapper(nn.LSTM(p.token_dim, p.hidden_dim, p.num_layers, batch_first=True,
                                            dropout=p.dropout if p.num_layers > 1 else 0.))
    if p.predictor == "quant":
        pred = QuantTokenPredictor(num_toks, p.token_dim, shared_embedding=emb.weight, quant_criterion=p.quant_criterion)
    elif p.predictor == "mos":   # MoS by default
        pred = MoSProjection(p.num_mixture, p.hidden_dim, num_toks)
    else: # tied
        pred = nn.Linear(p.token_dim, num_toks, bias=False)    # logits output
        pred.weight = emb.weight

    if p.embedding_norm:
        emb = nn.Sequential(emb, Normalization())

    return SeqModeling(embedding=emb, encoder=encoder, padding=0, prediction=pred, attention=None)

class CFQTrainingUpdater(TrainingUpdater):
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

def main():
    args = setup(seed="2021", dataset="cfq_mcd1", translator="cfq")
    bot = TrialBot(trial_name="lstm_lm", get_model_func=get_model, args=args)

    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def training_metrics(bot: TrialBot):
        bot.logger.info("Epoch Metrics:")
        bot.logger.info(json.dumps(bot.model.get_metric(reset=True)))

    from utils.trial_bot_extensions import print_hyperparameters
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 100)
    if not args.test:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import debug_models, end_with_nan_loss, evaluation_on_dev_every_epoch

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 120)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)
        bot.add_event_handler(Events.STARTED, debug_models, 100)
        bot.updater = CFQTrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()
