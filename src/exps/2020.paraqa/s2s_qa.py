import sys
import logging

from trialbot.training import TrialBot
from trialbot.training import Registry
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from utils.root_finder import find_root
from trialbot.utils.move_to_device import move_to_device
from models.base_s2s.base_seq2seq import BaseS2SBuilder

import datasets.complex_web_q
import datasets.complex_web_q_translator

ROOT = find_root()

@Registry.hparamset()
def question_to_answer():
    from trialbot.training.hparamset import HyperParamSet
    p = HyperParamSet.common_settings(ROOT)
    p.TRAINING_LIMIT = 50
    p.batch_sz = 128

    p.emb_sz = 256
    p.src_namespace = 'ns_q'
    p.tgt_namespace = 'ns_lf'
    p.hidden_sz = 128
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "dot_product"
    p.concat_attn_to_dec_input = True
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.dropout = .2
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    return p

@Registry.hparamset()
def machine_question_to_answer():
    p = question_to_answer()
    p.src_namespace = 'ns_mq'
    return p

class CompWebQTrainingUpdater(TrainingUpdater):
    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()
        batch = next(iterator)
        q = batch['q']
        mq = batch['mq']
        sparql = batch['sparql']

        if device >= 0:
            q = move_to_device(q, device)
            mq = move_to_device(mq, device)
            sparql = move_to_device(sparql, device)

        output = model(source_tokens=(q if self.src_ns == 'ns_q' else mq), target_tokens=sparql)
        if not self._dry_run:
            loss = output['loss']
            loss.backward()
            optim.step()
        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'CompWebQTrainingUpdater':
        updater = super().from_bot(bot)
        updater.src_ns = bot.hparams.src_namespace
        return updater

class CompWebQTestingUpdater(TestingUpdater):
    def update_epoch(self):
        model, iterator = self._models[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        batch = next(iterator)
        q = batch['q']
        mq = batch['mq']
        sparql = batch['sparql']

        if device >= 0:
            q = move_to_device(q, device)
            mq = move_to_device(mq, device)
            sparql = move_to_device(sparql, device)

        output = model(source_tokens=(q if self.src_ns == 'ns_q' else mq))
        output = model.decode(output)

        model.compute_metric(output['predictions'], sparql)
        output["qid"] = batch['qid']
        output["_raw"] = batch["_raw"]
        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'CompWebQTestingUpdater':
        updater = super().from_bot(bot)
        updater.src_ns = bot.hparams.src_namespace
        return updater

def setup():
    from trialbot.training import TrialBot
    import sys
    args = sys.argv[1:]
    args += ['--seed', '2020']
    if '--dataset' not in sys.argv:
        args += ['--dataset', 'CompWebQ']
    if '--translator' not in sys.argv:
        args += ['--translator', 'CompWebQTranslator']

    parser = TrialBot.get_default_parser()
    parser.add_argument('--dev', action="store_true")
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

    return args

def main():
    args = setup()

    bot = TrialBot(trial_name="seq2seq", get_model_func=BaseS2SBuilder.from_param_and_vocab, args=args)

    from trialbot.training import Events
    if args.test:
        from trialbot.training.trial_bot import Engine
        new_engine = Engine()
        new_engine.register_events(*Events)
        bot._engine = new_engine

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def print_output(bot: TrialBot):
            import json
            output = bot.state.output
            if output is None:
                return

            for qid, raw, pred in zip(*map(output.get, ("qid", "_raw", "predicted_tokens"))):
                to_print = {
                    "qid": qid,
                    "src": raw['q_toks'] if bot.hparams.src_namespace == 'ns_q' else raw['mq_toks'],
                    "gold": raw['sparql_toks'],
                    "pred": pred
                }
                print(json.dumps(to_print))

        @bot.attach_extension(Events.EPOCH_COMPLETED)
        def final_metric(bot: TrialBot):
            import json
            metric = bot.model.get_metrics()
            print(json.dumps(metric))

        bot.updater = CompWebQTestingUpdater.from_bot(bot)
    else:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import debug_models, end_with_nan_loss

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.STARTED, debug_models, 100)
        bot.updater = CompWebQTrainingUpdater.from_bot(bot)
    bot.run()

if __name__ == '__main__':
    main()
