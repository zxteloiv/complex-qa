import sys
sys.path.insert(0, '..')
from typing import List, Generator, Tuple, Mapping, Optional
import logging
import torch.nn
import numpy as np
import random

from trialbot.data import NSVocabulary, PADDING_TOKEN
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.training.hparamset import HyperParamSet
from trialbot.data import Iterator, RandomIterator
from trialbot.utils.move_to_device import move_to_device

from utils.root_finder import find_root
_ROOT = find_root()

@Registry.hparamset()
def atis_five_giant():
    p = HyperParamSet.common_settings(_ROOT)
    p.encoder_kernel_size = 3
    p.num_classes = 2     # either 0 (true) or 1 (false), only 2 classes
    p.fusion = "full"         # simple, full
    p.alignment = "linear"    # identity, linear, bilinear
    p.connection = "aug"      # none, residual, aug
    p.prediction = "full"     # simple, full, symmetric
    p.encoder = "bilstm"
    p.emb_sz = 256
    p.hidden_size = 128
    p.num_stacked_block = 2
    p.num_stacked_encoder = 2
    p.dropout = .2
    p.TRAINING_LIMIT = 50
    p.batch_sz = 128
    return p

@Registry.hparamset()
def atis_giant_finetune():
    p = atis_five_giant()
    p.batch_sz = 32
    return p

@Registry.hparamset()
def atis_giant_v2():
    p = atis_five_giant()
    p.alignment = "bilinear"    # identity, linear, bilinear
    p.prediction = "full"     # simple, full, symmetric
    p.encoder = "bilstm"
    p.emb_sz = 256
    p.hidden_size = 128
    p.num_stacked_block = 2
    p.num_stacked_encoder = 2
    p.dropout = .2
    p.TRAINING_LIMIT = 50
    p.batch_sz = 64
    return p

# @Registry.hparamset()
# def atis_giant_v3():
#     p = atis_five_giant()
#     p.alignment = "bilinear"    # identity, linear, bilinear
#     p.prediction = "full"     # simple, full, symmetric
#     p.encoder = "bilstm"
#     p.emb_sz = 300
#     p.hidden_size = 150
#     p.num_stacked_block = 1
#     p.num_stacked_encoder = 2
#     p.dropout = .2
#     p.TRAINING_LIMIT = 50
#     p.batch_sz = 64
#     return p
#
# @Registry.hparamset()
# def atis_giant_v4():
#     p = atis_giant_v3()
#     p.num_stacked_encoder = 1
#     return p

@Registry.hparamset()
def atis_neo_giant():
    p = atis_giant_v2()
    p.alignment = "bilinear"    # identity, linear, bilinear
    p.prediction = "full"     # simple, full, symmetric
    p.encoder = "bilstm"
    p.fusion = "neo"
    p.emb_sz = 256
    p.hidden_size = 128
    p.num_stacked_block = 2
    p.num_stacked_encoder = 2
    p.dropout = .2
    p.TRAINING_LIMIT = 200
    p.batch_sz = 64
    return p

# @Registry.hparamset()
# def django_fifteen_giant():
#     hparams = atis_five_giant()
#     hparams.TRAINING_LIMIT = 20
#     return hparams

@Registry.hparamset()
def django_neo_giant():
    p = atis_neo_giant()
    p.TRAINING_LIMIT = 100
    return p

import datasets.atis_rank
import datasets.atis_rank_translator
import datasets.django_rank
import datasets.django_rank_translator

def get_model(hparams, vocab):
    from experiments.build_model import get_giant_model
    return get_giant_model(hparams, vocab)

class DevTuningUpdater(TrainingUpdater):
    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        device = self._device
        batch = next(iterator)
        # eid, hyp_rank, sent_a, sent_b = list(map(batch.get, ("ex_id", "hyp_rank", "source_tokens", "hyp_tokens")))
        sent_a, sent_b, label = list(map(batch.get, ("source_tokens", "hyp_tokens", "hyp_label")))
        if iterator.is_new_epoch:
            self.stop_epoch()

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)
            label = move_to_device(label, device)

        model.train()
        scores = model.inference(sent_a, sent_b)
        optim.zero_grad()
        final_score = model.forward_loss_weight(*scores)
        loss = torch.nn.functional.smooth_l1_loss(final_score, label.float())
        loss.backward()
        optim.step()
        return {"loss": loss}

    @classmethod
    def from_bot(cls, bot: TrialBot):
        from models.matching.giant_ranker import GiantRanker
        model: GiantRanker = bot.model
        optim = torch.optim.AdamW([model.loss_weighting])
        iterator = RandomIterator(bot.dev_set, bot.hparams.batch_sz, bot.translator, shuffle=True, repeat=True)
        obj = cls(model, iterator, optim, bot.args.device)
        return obj

class GiantTrainingUpdater(TrainingUpdater):
    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()

        batch = next(iterator)
        sent_a, sent_b, label = list(map(batch.get, ("source_tokens", "hyp_tokens", "hyp_label")))

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)
            label = move_to_device(label, device)

        loss = model(sent_a, sent_b, label)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), self._grad_clip_val)

        # do some clipping
        if torch.isnan(loss).any():
            logging.getLogger().error("NaN loss encountered")
        else:
            optim.step()
        return {"loss": loss}

    @classmethod
    def from_bot(cls, bot: TrialBot):
        obj: TrainingUpdater = super().from_bot(bot)
        del obj._optims

        args, hparams, model = bot.args, bot.hparams, bot.model
        bot.logger.info("Changed to AdamW with not only the same lr, beta, but also the default AdamW weight decay")
        optim = torch.optim.AdamW(model.parameters(), hparams.ADAM_LR, hparams.ADAM_BETAS)
        obj._optims = [optim]

        obj._grad_clip_val = hparams.GRAD_CLIPPING
        return obj

class GiantTestingUpdater(TestingUpdater):
    def update_epoch(self):
        model, iterator, device = self._models[0], self._iterators[0], self._device
        model.eval()
        batch = next(iterator)
        eid, hyp_rank, sent_a, sent_b = list(map(batch.get, ("ex_id", "hyp_rank", "source_tokens", "hyp_tokens")))
        if iterator.is_new_epoch:
            self.stop_epoch()

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)

        scores = model.inference(sent_a, sent_b)
        correct_score = model.forward_loss_weight(*scores)
        return {"ranking_score": correct_score, "ex_id": eid, "hyp_rank": hyp_rank}

def main():
    import sys
    args = sys.argv[1:]
    args += ['--seed', '2020']
    if '--dataset' not in sys.argv:
        args += ['--dataset', 'atis_five_hyp']
    if '--translator' not in sys.argv:
        args += ['--translator', 'atis_rank']

    parser = TrialBot.get_default_parser()
    parser.add_argument('--fine-tune', action="store_true")
    args = parser.parse_args(args)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    bot = TrialBot(trial_name="baseline2_giant", get_model_func=get_model, args=args)
    bot.translator.turn_special_token(on=True)
    if args.test:
        import trialbot
        new_engine = trialbot.training.trial_bot.Engine()
        new_engine.register_events(*Events)
        bot._engine = new_engine

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def print_output(bot: TrialBot):
            import json
            output = bot.state.output
            if output is None:
                return

            output_keys = ("ex_id", "hyp_rank", "ranking_score")
            for eid, hyp_rank, score in zip(*map(output.get, output_keys)):
                print(json.dumps(dict(zip(output_keys, (eid, hyp_rank, score.item())))))

        bot.updater = GiantTestingUpdater.from_bot(bot)
    else:
        from trialbot.training.extensions import every_epoch_model_saver
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import debug_models, end_with_nan_loss

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.STARTED, debug_models, 100)
        if args.fine_tune:
            bot.updater = DevTuningUpdater.from_bot(bot)
        else:
            bot.updater = GiantTrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()


