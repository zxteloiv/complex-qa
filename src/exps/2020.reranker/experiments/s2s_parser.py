from os.path import join, abspath, dirname
import sys
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..', '..')))   # up to src
from trialbot.training import TrialBot, Registry


def main():
    from utils.trialbot.trialbot_setup import setup
    args = setup(seed=2020, translator='top', dataset='top', hparamset='s2s_top')
    bot = TrialBot(trial_name='s2s_parser', get_model_func=get_model, args=args)
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
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90,
                              rewrite_eval_hparams={"batch_sz": 32},
                              interval=1,
                              skip_first_epochs=15,
                              )

        @bot.attach_extension(Events.EPOCH_COMPLETED, 95)
        def collect_garbage(bot: TrialBot):
            for optim in bot.updater._optims:
                optim.zero_grad()

            if bot.state.output is not None:
                del bot.state.output
            import gc
            gc.collect()
            if bot.args.device >= 0:
                import torch.cuda
                torch.cuda.empty_cache()

        bot.add_event_handler(Events.EPOCH_STARTED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 95)

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

            model = bot.model
            output = model.revert_tensor_to_string(output)

            batch_print = []
            for src, gold, pred in zip(*map(output.get, ("source_tokens", "target_tokens", "predicted_tokens"))):
                to_print = f'SRC:  {" ".join(src)}\nGOLD: {" ".join(gold)}\nPRED: {" ".join(pred)}'
                batch_print.append(to_print)

            sep = '\n' + '-' * 60 + '\n'
            print(sep.join(batch_print))

    bot.run()

def get_model(p, vocab):
    if getattr(p, 'model_arch', 'rnn') == 'rnn':
        from models.base_s2s.base_seq2seq import BaseSeq2Seq
        return BaseSeq2Seq.from_param_and_vocab(p, vocab)
    else:
        from models.transformer.parallel_seq2seq import ParallelSeq2Seq
        return ParallelSeq2Seq.from_param_and_vocab(p, vocab)


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

@Registry.hparamset()
def s2s_top_unified():
    p = s2s_top()
    p.src_namespace = 'unified_vocab'
    p.tgt_namespace = 'unified_vocab'
    return p

@Registry.hparamset()
def s2s_top_unified_aggressive():
    p = s2s_top_unified()
    p.emb_sz = 300
    p.hidden_sz = 300
    p.dropout = .5
    return p

@Registry.hparamset()
def top_transformer_unified():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 100  # in num of epochs
    p.WEIGHT_DECAY = .2
    p.ADAM_LR = 1e-3
    p.OPTIM = "RAdam"
    p.batch_sz = 128
    p.emb_sz = 256
    p.src_namespace = 'unified_vocab'
    p.tgt_namespace = 'unified_vocab'

    p.model_arch = 'transformer'    # transformer, universal_transformer, or rnn

    p.num_enc_layers = 4
    p.num_dec_layers = 4
    p.dropout = .2
    p.attention_dropout = .05
    p.num_heads = 8
    p.max_decoding_len = 100
    p.nonlinear_activation = "mish"

    p.predictor = 'quant'  # mos, quant
    # used for quant predictor
    p.tied_tgt_predictor = False
    p.quant_crit = "projection"  # distance, projection, dot_product
    # used for mos predictor
    # p.num_mixture = 10

    p.beam_size = 1
    p.diversity_factor = 0.
    p.acc_factor = 1.

    p.flooding_bias = 0.2
    return p

if __name__ == '__main__':
    main()