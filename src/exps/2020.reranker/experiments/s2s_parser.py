from os.path import join, abspath, dirname
import sys
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..', '..')))   # up to src
from trialbot.training import TrialBot, Events, Registry
import datasets.top

def main():
    from utils.trialbot_setup import setup
    from models.base_s2s.base_seq2seq import BaseSeq2Seq
    args = setup(seed=2020, translator='top', dataset='top', hparamset='s2s_top')
    bot = TrialBot(trial_name='s2s_parser', get_model_func=BaseSeq2Seq.from_param_and_vocab, args=args)
    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def get_metrics(bot: TrialBot):
        import json
        print(json.dumps(bot.model.get_metric(reset=True)))

    from trialbot.training import Events
    if not args.test:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import end_with_nan_loss
        from utils.trial_bot_extensions import evaluation_on_dev_every_epoch
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        @bot.attach_extension(Events.STARTED)
        def print_models(bot: TrialBot):
            print(str(bot.models))

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

@Registry.hparamset()
def s2s_top():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 200  # in num of epochs
    p.OPTIM = "RAdam"
    p.emb_sz = 256
    p.src_namespace = 'tokenized_utterance'
    p.tgt_namespace = 'top_representation'
    p.hidden_sz = 256
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "dot_product"
    p.concat_attn_to_dec_input = False
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.dropout = .2
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    p.tied_decoder_embedding = True
    return p

if __name__ == '__main__':
    main()