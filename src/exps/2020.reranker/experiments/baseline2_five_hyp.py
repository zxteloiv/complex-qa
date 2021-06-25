import sys
sys.path.insert(0, '..')
import logging
import torch.nn
from fairseq.optim.adafactor import Adafactor

from trialbot.data import NSVocabulary
from trialbot.training import Registry, TrialBot, Events
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.move_to_device import move_to_device

from utils.root_finder import find_root
_ROOT = find_root()

@Registry.hparamset()
def atis_neo_five():
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
def django_neo_five():
    hparams = atis_neo_five()
    hparams.TRAINING_LIMIT = 60
    return hparams

@Registry.hparamset()
def atis_neo_five_dropout():
    p = atis_neo_five()
    p.dropout = .2
    p.discrete_dropout = .1
    p.TRAINING_LIMIT = 200
    return p

@Registry.hparamset()
def django_neo_five_dropout():
    p = atis_neo_five_dropout()
    p.TRAINING_LIMIT = 60
    return p

@Registry.hparamset()
def atis_deep_stack():
    p = atis_neo_five_dropout()
    p.num_stacked_block = 4
    p.num_stacked_encoder = 1
    p.TRAINING_LIMIT = 200
    return p

@Registry.hparamset()
def django_deep_stack():
    p = django_neo_five_dropout()
    p.num_stacked_block = 4
    p.num_stacked_encoder = 1
    p.TRAINING_LIMIT = 60
    return p

@Registry.hparamset()
def atis_deeper():
    p = atis_neo_five_dropout()
    p.num_stacked_block = 6
    p.num_stacked_encoder = 1
    p.TRAINING_LIMIT = 200
    return p

@Registry.hparamset()
def django_deeper():
    p = django_neo_five_dropout()
    p.num_stacked_block = 6
    p.num_stacked_encoder = 1
    p.TRAINING_LIMIT = 60
    return p


def get_model(hparams, vocab: NSVocabulary):
    from experiments.build_model import get_re2_char_model
    return get_re2_char_model(hparams, vocab)

class ChRE2TrainingUpdater(TrainingUpdater):
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

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)
            sent_char_a = move_to_device(sent_char_a, device)
            sent_char_b = move_to_device(sent_char_b, device)
            label = move_to_device(label, device)

        logits = model(sent_a, sent_char_a, sent_b, sent_char_b)
        if self._dry_run:
            return 0

        loss = torch.nn.functional.cross_entropy(logits, label)
        loss.backward()

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
        optim = Adafactor(model.parameters(), weight_decay=hparams.weight_decay)
        bot.logger.info("Use Adafactor optimizer: " + str(optim))
        obj._optims = [optim]
        return obj

class ChRE2TestingUpdater(TestingUpdater):
    def update_epoch(self):
        model, iterator, device = self._models[0], self._iterators[0], self._device
        model.eval()
        batch = next(iterator)
        if iterator.is_new_epoch:
            self.stop_epoch()

        eid = batch['ex_id']
        hyp_rank = batch['hyp_rank']
        sent_a = batch['source_tokens']
        sent_b = batch['hyp_tokens']
        sent_char_a = batch['src_char_ids']
        sent_char_b = batch['hyp_char_ids']

        if device >= 0:
            sent_a = move_to_device(sent_a, device)
            sent_b = move_to_device(sent_b, device)
            sent_char_a = move_to_device(sent_char_a, device)
            sent_char_b = move_to_device(sent_char_b, device)

        output = model(sent_a, sent_char_a, sent_b, sent_char_b)
        output = torch.log_softmax(output, dim=-1)
        correct_score = output[:, 1]
        return {"prediction": output, "ranking_score": correct_score, "ex_id": eid, "hyp_rank": hyp_rank}

def main():
    import sys
    args = sys.argv[1:]
    args += ['--seed', '2020']
    if '--dataset' not in sys.argv:
        args += ['--dataset', 'atis_five_hyp']
    if '--translator' not in sys.argv:
        args += ['--translator', 'atis_rank_char']

    parser = TrialBot.get_default_parser()
    args = parser.parse_args(args)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if hasattr(args, "seed") and args.seed:
        from trialbot.utils.fix_seed import fix_seed
        logging.info(f"set seed={args.seed}")
        fix_seed(args.seed)

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

        bot.updater = ChRE2TestingUpdater.from_bot(bot)
    else:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import debug_models, end_with_nan_loss

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.STARTED, debug_models, 100)
        bot.updater = ChRE2TrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()


