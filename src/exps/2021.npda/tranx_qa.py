from os.path import join, abspath, dirname, expanduser
import sys
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..')))   # up to src
from trialbot.training import TrialBot, Events, Registry, Updater

import logging
import datasets.cfq
import datasets.cfq_translator
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_sql_qa_datasets(Registry._datasets)
import datasets.cg_bundle_translator

def get_tranx_updater(bot: TrialBot):
    args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
    from utils.select_optim import select_optim
    from utils.maybe_random_iterator import MaybeRandomIterator

    params = model.parameters()
    optim = select_optim(p, params)
    logger.info(f"Using Optimizer {optim}")

    device, dry_run = args.device, args.dry_run
    repeat_iter = shuffle_iter = not args.debug
    iterator = MaybeRandomIterator(bot.train_set, p.batch_sz, bot.translator, shuffle=shuffle_iter, repeat=repeat_iter)
    if args.debug and args.skip:
        iterator.reset(args.skip)

    lr_scheduler_kwargs = getattr(p, 'lr_scheduler_kwargs', None)
    if lr_scheduler_kwargs is not None:
        from allennlp.training.learning_rate_schedulers import NoamLR
        lr_scheduler = NoamLR(optimizer=optim, **lr_scheduler_kwargs)
        bot.scheduler = lr_scheduler
        logger.info(f'the scheduler enabled: {lr_scheduler}')
        def _sched_step(bot: TrialBot):
            bot.scheduler.step_batch()
            if bot.state.iteration % 100 == 0:
                logger.info(f"update lr to {bot.scheduler.get_values()}")
        bot.add_event_handler(Events.ITERATION_COMPLETED, _sched_step, 100)

    from trialbot.training.updater import TrainingUpdater
    updater = TrainingUpdater(model, iterator, optim, device, dry_run)
    return updater

def main():
    from utils.trialbot_setup import setup
    from models.base_s2s.base_seq2seq import BaseSeq2Seq
    args = setup(seed=2021, translator='cfq_tranx_mod_ent_qa', dataset='cfq_mcd1', hparamset='cfq_mod_ent_tranx')
    bot = TrialBot(trial_name='tranx_qa', get_model_func=BaseSeq2Seq.from_param_and_vocab, args=args)

    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def get_metrics(bot: TrialBot):
        import json
        print(json.dumps(bot.model.get_metric(reset=True)))

    from utils.trial_bot_extensions import print_hyperparameters
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)

    from trialbot.training import Events
    if not args.test:
        # --------------------- Training -------------------------------
        @bot.attach_extension(Events.STARTED)
        def print_models(bot: TrialBot):
            print(str(bot.models))

        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import end_with_nan_loss
        from utils.trial_bot_extensions import evaluation_on_dev_every_epoch, collect_garbage
        bot.add_event_handler(Events.EPOCH_STARTED, collect_garbage, 95)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90, rewrite_eval_hparams={"batch_sz": 32})
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 95)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)

        bot.updater = get_tranx_updater(bot)
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

@Registry.hparamset()
def cfq_mod_ent_tranx():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 10  # in num of epochs
    p.OPTIM = "adabelief"
    p.batch_sz = 32

    p.emb_sz = 256
    p.src_namespace = 'questionPatternModEntities'
    p.tgt_namespace = 'modent_rule_seq'
    p.hidden_sz = 256
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "dot_product"
    p.dec_inp_composer = 'cat_mapping'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'mapping_add'
    p.proj_inp_comp_activation = 'tanh'
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.dropout = .2
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    p.tied_decoder_embedding = False
    p.src_emb_trained_file = "~/.glove/glove.6B.100d.txt.gz"
    return p

@Registry.hparamset()
def cfq_mod_ent_moderate_tranx():
    p = cfq_mod_ent_tranx()
    p.TRAINING_LIMIT = 10  # in num of epochs
    p.WEIGHT_DECAY = .1
    p.emb_sz = 128
    p.hidden_sz = 128
    p.dec_hist_attn = "dot_product"
    return p

@Registry.hparamset()
def common_sql_tranx():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())

    p.TRAINING_LIMIT = 100  # in num of epochs
    p.OPTIM = "adabelief"
    p.WEIGHT_DECAY = .1
    p.ADAM_BETAS = (0.9, 0.999)
    p.batch_sz = 16

    p.emb_sz = 128
    p.hidden_sz = 128
    p.src_namespace = 'sent'
    p.tgt_namespace = 'rule_seq'
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "none"
    p.dec_inp_composer = 'cat_mapping'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'cat_mapping'
    p.proj_inp_comp_activation = 'mish'
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.dropout = .5
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    p.tied_decoder_embedding = True
    return p

@Registry.hparamset()
def scholar_common():
    p = common_sql_tranx()
    p.TRAINING_LIMIT = 150
    p.lr_scheduler_kwargs = {"model_size": 400, "warmup_steps": 50} # noam lr_scheduler
    p.tied_decoder_embedding = False
    p.num_enc_layers = 1
    p.num_dec_layers = 1
    p.emb_sz = 100

    p.encoder = 'lstm'
    p.enc_out_dim = 300
    p.enc_attn = "dot_product"
    p.dec_in_dim = p.enc_out_dim
    p.dec_out_dim = p.enc_out_dim

    p.proj_in_dim = p.emb_sz

    p.enc_dropout = 0
    p.dec_dropout = 0.5
    p.dec_hist_attn = "none"
    p.dec_inp_composer = 'cat_mapping'
    p.dec_inp_comp_activation = 'relu'
    p.proj_inp_composer = 'cat_mapping'
    p.proj_inp_comp_activation = 'relu'
    p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"
    return p

@Registry.hparamset()
def atis_common():
    p = common_sql_tranx()
    p.TRAINING_LIMIT = 100
    p.tied_decoder_embedding = False
    p.num_enc_layers = 1
    p.num_dec_layers = 1
    p.emb_sz = 100
    p.hidden_sz = 300
    p.lr_scheduler_kwargs = {"model_size": 600, "warmup_steps": 50} # noam lr_scheduler
    p.WEIGHT_DECAY = 0.
    return p

@Registry.hparamset()
def advising_common():
    p = atis_common()
    return p

if __name__ == '__main__':
    main()
