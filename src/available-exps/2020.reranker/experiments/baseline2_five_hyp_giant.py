import sys
sys.path.insert(0, '..')
from typing import List, Generator, Tuple, Mapping, Optional
import logging
import torch.nn
import numpy as np
import random
from fairseq.optim.adafactor import Adafactor

from trialbot.data import NSVocabulary, PADDING_TOKEN
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.training.hparamset import HyperParamSet
from trialbot.data import Iterator, RandomIterator
from trialbot.utils.move_to_device import move_to_device

from utils.root_finder import find_root
_ROOT = find_root()

@Registry.hparamset()
def atis_giant_five():
    p = HyperParamSet.common_settings(_ROOT)
    p.alignment = "bilinear"    # identity, linear, bilinear
    p.prediction = "full"     # simple, full, symmetric
    p.encoder = "bilstm"
    p.pooling = "neo"  # neo or vanilla
    p.fusion = "neo"    # neo or vanilla
    p.connection = "aug"      # none, residual, aug
    p.num_classes = 2     # either 0 (true) or 1 (false), only 2 classes
    p.emb_sz = 256
    p.hidden_size = 256
    p.num_stacked_block = 2
    p.num_stacked_encoder = 2
    p.dropout = .5
    p.TRAINING_LIMIT = 200
    p.weight_decay = 0.2
    p.batch_sz = 64
    p.char_emb_sz = 128
    p.char_hid_sz = 128
    return p

@Registry.hparamset()
def django_giant_five():
    hparams = atis_giant_five()
    hparams.TRAINING_LIMIT = 60
    return hparams

@Registry.hparamset()
def atis_giant_five_dropout():
    p = atis_giant_five()
    p.dropout = .2
    p.discrete_dropout = .1
    p.TRAINING_LIMIT = 200
    return p

@Registry.hparamset()
def django_giant_five_dropout():
    p = atis_giant_five_dropout()
    p.TRAINING_LIMIT = 60
    return p

@Registry.hparamset()
def atis_deep_giant():
    p = atis_giant_five_dropout()
    p.num_stacked_block = 4
    p.num_stacked_encoder = 1
    p.TRAINING_LIMIT = 200
    return p

@Registry.hparamset()
def django_deep_giant():
    p = django_giant_five_dropout()
    p.num_stacked_block = 4
    p.num_stacked_encoder = 1
    p.TRAINING_LIMIT = 60
    return p

import datasets.atis_rank
import datasets.atis_rank_translator
import datasets.django_rank
import datasets.django_rank_translator

def get_model(hparams, vocab):
    from experiments.build_model import get_char_giant
    return get_char_giant(hparams, vocab)

class GiantTrainingUpdater(TrainingUpdater):
    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()

        batch = next(iterator)
        sent_a = batch['source_tokens']
        sent_b = batch['hyp_tokens']
        sent_char_a = batch['src_char_ids']
        sent_char_b = batch['hyp_char_ids']
        label = batch['hyp_label']
        rank = batch['hyp_rank']

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)
            sent_char_a = move_to_device(sent_char_a, device)
            sent_char_b = move_to_device(sent_char_b, device)
            label = move_to_device(label, device)
            rank = move_to_device(rank, device)

        loss = model(sent_a, sent_b, sent_char_a, sent_char_b, label, rank)
        loss.backward()

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
        optim = Adafactor(model.parameters(), weight_decay=hparams.weight_decay)
        bot.logger.info("Use Adafactor optimizer: " + str(optim))
        obj._optims = [optim]
        return obj

class GiantTestingUpdater(TestingUpdater):
    def update_epoch(self):
        model, iterator, device = self._models[0], self._iterators[0], self._device
        model.eval()
        batch = next(iterator)
        if iterator.is_new_epoch:
            self.stop_epoch()

        eid = batch['ex_id']
        rank = batch['hyp_rank']
        sent_a = batch['source_tokens']
        sent_b = batch['hyp_tokens']
        sent_char_a = batch['src_char_ids']
        sent_char_b = batch['hyp_char_ids']

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)
            sent_char_a = move_to_device(sent_char_a, device)
            sent_char_b = move_to_device(sent_char_b, device)
            rank = move_to_device(rank, device)

        scores = model.inference(sent_a, sent_b, sent_char_a, sent_char_b, rank)
        correct_score = model.forward_loss_weight(*scores)
        return {"ranking_score": correct_score, "ex_id": eid, "hyp_rank": rank,
                "rank_match": scores[0], "rank_a2b": scores[1], "rank_b2a": scores[2],
                }

def main():
    import sys
    args = sys.argv[1:]
    args += ['--seed', '2020']
    if '--dataset' not in sys.argv:
        args += ['--dataset', 'atis_five_hyp']
    if '--translator' not in sys.argv:
        args += ['--translator', 'atis_rank']

    parser = TrialBot.get_default_parser()
    args = parser.parse_args(args)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if hasattr(args, "seed") and args.seed:
        from utils.fix_seed import fix_seed
        logging.info(f"set seed={args.seed}")
        fix_seed(args.seed)

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

            output_keys = ("ex_id", "hyp_rank", "ranking_score", "rank_match", "rank_a2b", "rank_b2a")
            for eid, hyp_rank, score, r_m, r_a2b, r_b2a in zip(*map(output.get, output_keys)):
                print(json.dumps(dict(zip(output_keys, (eid, hyp_rank.item(), score.item(),
                                                        r_m.item(), r_a2b.item(), r_b2a.item())))))

        bot.updater = GiantTestingUpdater.from_bot(bot)
    else:
        from trialbot.training.extensions import every_epoch_model_saver
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import debug_models, end_with_nan_loss

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.STARTED, debug_models, 100)
        bot.updater = GiantTrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()


