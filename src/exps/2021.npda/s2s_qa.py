import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import logging

from trialbot.training import TrialBot
from trialbot.training import Registry
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.utils.move_to_device import move_to_device
from models.base_s2s.base_seq2seq import BaseS2SBuilder, BaseSeq2Seq

import datasets.cfq
import datasets.cfq_translator

@Registry.hparamset()
def cfq_seq_qa():
    from trialbot.training.hparamset import HyperParamSet
    from utils.root_finder import find_root
    ROOT = find_root()
    p = HyperParamSet.common_settings(ROOT)
    p.TRAINING_LIMIT = 50
    p.batch_sz = 128
    p.weight_decay = .2

    p.emb_sz = 256
    p.src_namespace = 'questionPatternModEntities'
    p.tgt_namespace = 'sparqlPatternModEntities'
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

class RAdamTrainingUpdater(TrainingUpdater):
    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'RAdamTrainingUpdater':
        updater = super().from_bot(bot)
        del updater._optims
        args, hparams, model = bot.args, bot.hparams, bot.model
        from radam import RAdam
        optim = RAdam(model.parameters(), lr=hparams.ADAM_LR, weight_decay=hparams.weight_decay)
        bot.logger.info("Use RAdam optimizer: " + str(optim))
        updater._optims = [optim]
        return updater

def main():
    from utils.trialbot_setup import setup
    args = setup(seed=2021)

    bot = TrialBot(trial_name="s2s_qa", get_model_func=BaseS2SBuilder.from_param_and_vocab, args=args)

    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def get_metrics(bot: TrialBot):
        import json
        print(json.dumps(bot.model.get_metric(reset=True)))

    from trialbot.training import Events
    if not args.test:
        # --------------------- Training -------------------------------
        bot.updater = RAdamTrainingUpdater.from_bot(bot)
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import debug_models, end_with_nan_loss
        from utils.trial_bot_extensions import evaluation_on_dev_every_epoch
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.STARTED, debug_models, 100)

        # debug strange errors by inspecting running time, data size, etc.
        if args.debug:
            from utils.trial_bot_extensions import track_pytorch_module_forward_time
            bot.add_event_handler(Events.STARTED, track_pytorch_module_forward_time, 100)

            @bot.attach_extension(Events.ITERATION_COMPLETED)
            def batch_data_size(bot: TrialBot):
                output = bot.state.output
                if output is None:
                    return

                print("source_size:", output['source'].size(),
                      "target_size:", output['target'].size(),
                      "pred_size:", output['predictions'].size())
    elif args.debug:
        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def print_output(bot: TrialBot):
            import json
            output = bot.state.output
            if output is None:
                return

            model: BaseSeq2Seq = bot.model
            output = model.revert_tensor_to_string(output)

            batch_print = []
            for src, gold, pred in zip(*map(output.get, ("source_tokens", "target_tokens", "predicted_tokens"))):
                to_print = f'SRC:  {" ".join(src)}\nGOLD: {" ".join(gold)}\nPRED: {" ".join(pred)}'
                batch_print.append(to_print)

            sep = '\n' + '-' * 60 + '\n'
            print(sep.join(batch_print))
    bot.run()

if __name__ == '__main__':
    main()
