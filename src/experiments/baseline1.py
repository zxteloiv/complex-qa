import sys
sys.path.insert(0, '..')
import logging
import torch.nn
import random

from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.move_to_device import move_to_device

from utils.root_finder import find_root
_ROOT = find_root()

@Registry.hparamset()
def atis_none():
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

import datasets.atis_rank
import datasets.atis_rank_translator

def get_model(hparams, vocab):
    from experiments.build_model import get_re2_model
    return get_re2_model(hparams, vocab)

class Re2TrainingUpdater(TrainingUpdater):
    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()

        batch = next(iterator)
        sent_a, sent_b = list(map(batch.get, ("source_tokens", "target_tokens")))

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)

        batch_size = sent_a.size()[0]
        pos_target = sent_a.new_ones((batch_size,))
        pred_for_pos = model(sent_a, sent_b)
        if self._dry_run:
            return 0

        loss1 = torch.nn.functional.cross_entropy(pred_for_pos, pos_target)
        loss1.backward()

        neg_target = sent_a.new_zeros((batch_size,))
        pred_for_neg = model(sent_a, torch.roll(sent_b, random.randrange(1, batch_size), dims=0))
        loss2 = torch.nn.functional.cross_entropy(pred_for_neg, neg_target)
        loss2.backward()
        optim.step()

        loss = loss1 + loss2
        return {"loss": loss, "prediction": pred_for_pos, "ranking_score": pred_for_pos[:, 1]}

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
        args += ['--dataset', 'atis_none_hyp']
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

    bot = TrialBot(trial_name="reranking_baseline1", get_model_func=get_model, args=args)
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

        bot.updater = Re2TestingUpdater.from_bot(bot)
    else:
        from trialbot.training.extensions import every_epoch_model_saver

        def output_inspect(bot: TrialBot, keys):
            iteration = bot.state.iteration
            if iteration % 4 != 0:
                return

            output = bot.state.output
            bot.logger.info(", ".join(f"{k}={v}" for k, v in zip(keys, map(output.get, keys))))

        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        # bot.add_event_handler(Events.ITERATION_COMPLETED, output_inspect, 100, keys=["loss"])
        bot.updater = Re2TrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()

