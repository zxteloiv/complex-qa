import sys
sys.path.insert(0, '..')
from typing import List, Generator, Tuple, Mapping, Optional
import logging
import torch.nn
import numpy as np
import random
from models.matching.re2 import RE2

from trialbot.data import NSVocabulary
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.move_to_device import move_to_device

from utils.root_finder import find_root
_ROOT = find_root()

@Registry.hparamset()
def atis_five():
    hparams = HyperParamSet.common_settings(_ROOT)
    hparams.emb_sz = 300
    hparams.hidden_size = 150
    hparams.encoder_kernel_size = 3
    hparams.num_classes = 2     # either 0 (true) or 1 (false), only 2 classes
    hparams.num_stacked_block = 3
    hparams.num_stacked_encoder = 2
    hparams.dropout = .5
    hparams.fusion = "full"         # simple, full
    hparams.alignment = "linear"    # identity, linear
    hparams.connection = "aug"      # none, residual, aug
    hparams.prediction = "full"     # simple, full, symmetric
    return hparams

@Registry.hparamset()
def atis_five_v2():
    hparams = atis_five()
    hparams.hidden_size = 200
    hparams.num_stacked_encoder = 3
    hparams.prediction = "symmetric"
    return hparams

@Registry.hparamset()
def atis_five_v3():
    hparams = atis_five()
    hparams.emb_sz = 256
    hparams.hidden_size = 128
    hparams.batch_sz = 32
    hparams.prediction = "symmetric"
    return hparams

@Registry.hparamset()
def atis_five_big():
    hparams = atis_five()
    hparams.num_stacked_block = 5
    return hparams

@Registry.hparamset()
def atis_five_bigger():
    hparams = atis_five()
    hparams.num_stacked_block = 7
    return hparams

import datasets.atis_rank
import datasets.atis_rank_translator

def get_model(hparams, vocab: NSVocabulary):
    model = RE2.get_model(emb_sz=hparams.emb_sz,
                          num_tokens_a=vocab.get_vocab_size('nl'),
                          num_tokens_b=vocab.get_vocab_size('lf'),
                          hid_sz=hparams.hidden_size,
                          enc_kernel_sz=hparams.encoder_kernel_size,
                          num_classes=hparams.num_classes,
                          num_stacked_blocks=hparams.num_stacked_block,
                          num_encoder_layers=hparams.num_stacked_encoder,
                          dropout=hparams.dropout,
                          fusion_mode=hparams.fusion,
                          alignment_mode=hparams.alignment,
                          connection_mode=hparams.connection,
                          prediction_mode=hparams.prediction,
                          use_shared_embedding=False,
                          )
    return model

class Re2TrainingUpdater(TrainingUpdater):
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

        logits = model(sent_a, sent_b)
        if self._dry_run:
            return 0

        loss = torch.nn.functional.cross_entropy(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), self._grad_clip_val)

        # do some clipping
        if torch.isnan(loss).any():
            logging.getLogger().error("NaN loss encountered")

        else:
            optim.step()

        return {"loss": loss, "prediction": logits}

    @classmethod
    def from_bot(cls, bot: TrialBot):
        obj: TrainingUpdater = super().from_bot(bot)
        del obj._optims

        args, hparams, model = bot.args, bot.hparams, bot.model
        bot.logger.info("Changed to AdamW with not only the same lr, beta, but also the default AdamW weight decay")
        optim = torch.optim.Adam(model.parameters(), hparams.ADAM_LR, hparams.ADAM_BETAS)
        obj._optims = [optim]

        obj._grad_clip_val = hparams.GRAD_CLIPPING
        return obj

class Re2TestingUpdater(TestingUpdater):
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

        output = model(sent_a, sent_b)
        output = torch.log_softmax(output, dim=-1)
        correct_score = output[:, 1]
        return {"prediction": output, "ranking_score": correct_score, "ex_id": eid, "hyp_rank": hyp_rank}

def main():
    import sys
    args = sys.argv[1:]
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

    bot = TrialBot(trial_name="reranking_baseline2", get_model_func=get_model, args=args)
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

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def end_with_nan_loss(bot: TrialBot):
            output = bot.state.output
            if output is None:
                return
            loss = output["loss"]
            def _isnan(x):
                if isinstance(x, torch.Tensor):
                    return bool(torch.isnan(x).any())
                elif isinstance(x, np.ndarray):
                    return bool(np.isnan(x).any())
                else:
                    import math
                    return math.isnan(x)

            if _isnan(loss):
                bot.logger.error("NaN loss encountered, training ended")
                bot.state.epoch = bot.hparams.TRAINING_LIMIT + 1

        bot.updater = Re2TestingUpdater.from_bot(bot)
    else:
        from trialbot.training.extensions import every_epoch_model_saver
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.updater = Re2TrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()


