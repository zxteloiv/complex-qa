from trialbot.training import Registry, TrialBot, Events
import argparse
from os import path as osp


def main():
    from utils.trialbot.setup_cli import setup as setup_cli
    import datasets.comp_gen_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    cg_bundle.install_cross_domain_parsed_qa_datasets(Registry._datasets)
    cg_bundle.install_raw_qa_datasets(Registry._datasets)
    cg_bundle.install_cross_domain_raw_qa_datasets(Registry._datasets)
    import datasets.cg_bundle_translator
    bot = setup_common_bot(setup_cli(seed=2021, device=0, translator='plm2s'))
    bot.run()


@Registry.hparamset()
def s2s():
    p = base_hparams()
    p.TRAINING_LIMIT = 400
    p.batch_sz = 16
    p.src_namespace = 'sent'
    p.tgt_namespace = 'sql'
    p.encoder = 'bilstm'
    return p


@Registry.hparamset()
def plm2s():
    p = s2s()
    p.lr_scheduler_kwargs = None
    p.src_namespace = None
    p.decoder_init_strategy = "avg_all"
    plm_path = osp.abspath(osp.expanduser('~/.cache/complex_qa/bert-base-uncased'))
    p.encoder = 'plm:' + plm_path
    p.TRANSLATOR_KWARGS = {"model_name": plm_path}
    return p


@Registry.hparamset()
def hungarian_reg():
    p = s2s()
    p.attn_supervision = 'hungarian_reg'
    return p


@Registry.hparamset()
def hungarian_xent():
    p = s2s()
    p.attn_supervision = 'hungarian_xent'
    return p


@Registry.hparamset()
def hungarian_reg_xent():
    p = s2s()
    p.attn_supervision = 'hungarian_reg_xent'
    return p


@Registry.hparamset()
def plm2s_hungarian_reg():
    p = plm2s()
    p.attn_supervision = 'hungarian_reg'
    return p


@Registry.hparamset()
def plm2s_hungarian_sup():
    p = plm2s()
    p.attn_supervision = 'hungarian_sup'
    return p


def get_updater(bot: TrialBot):
    args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
    from utils.select_optim import select_optim

    if p.encoder.startswith('plm:'):
        bert_params, nonbert_params = [], []
        for name, param in model.named_parameters():
            if 'pretrained_model' in name:  # the attribute is used by SeqPLMEmbedEncoder wrapper.
                bert_params.append(param)
            else:
                nonbert_params.append(param)

        params = [{'lr': 1e-5, 'params': bert_params}, {'params': nonbert_params}]
    else:
        params = model.parameters()

    optim = select_optim(p, params)
    logger.info(f"Using Optimizer {optim}")

    from trialbot.data.iterators import RandomIterator
    iterator = RandomIterator(len(bot.train_set), p.batch_sz)
    logger.info(f"Using RandomIterator: on volume={len(bot.train_set)} with batch={p.batch_sz}")

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

    from trialbot.training.updaters.training_updater import TrainingUpdater
    updater = TrainingUpdater(bot.train_set, bot.translator, model, iterator, optim, args.device, args.dry_run)
    return updater


def setup_common_bot(args: argparse.Namespace, get_model_func=None, trialname: str = None):
    assert args is not None
    from models.base_s2s.model_factory import Seq2SeqBuilder
    get_model_func = get_model_func or Seq2SeqBuilder.from_param_and_vocab
    trialname = trialname or f'enc2dec-{args.hparamset}'
    bot = TrialBot(trial_name=trialname, get_model_func=get_model_func, args=args)
    from utils.trialbot.setup_bot import setup_bot, add_metric_printing

    if not args.test:
        bot = setup_bot(bot, epoch_model_saving=False)
        from trialbot.training.extensions import every_epoch_model_saver
        if args.seed == 2021 or args.seed == '2021':
            bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.updater = get_updater(bot)
    else:
        bot = add_metric_printing(bot, 'Testing Metrics')

    return bot


def base_hparams():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 150
    p.WEIGHT_DECAY = 0.
    p.OPTIM = "adabelief"
    p.ADAM_BETAS = (0.9, 0.999)
    p.batch_sz = 16

    p.lr_scheduler_kwargs = {'model_size': 400, 'warmup_steps': 50}
    p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"

    p.hidden_sz = 300
    p.dropout = .5
    p.decoder = "lstm"
    p.max_decoding_step = 100
    p.scheduled_sampling = .1

    p.num_enc_layers = 1
    p.num_dec_layers = 1

    p.tied_decoder_embedding = False
    p.emb_sz = 100

    p.enc_out_dim = p.hidden_sz
    p.dec_in_dim = p.hidden_sz
    p.dec_out_dim = p.hidden_sz

    p.enc_dec_trans_usage = 'consistent'
    p.enc_dec_trans_act = 'mish'
    p.enc_dec_trans_forced = True

    p.proj_in_dim = p.emb_sz

    p.enc_dropout = 0
    p.dec_dropout = 0.5
    p.enc_attn = "dot_product"
    p.dec_hist_attn = "none"
    p.dec_inp_composer = 'cat_mapping'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'cat_mapping'
    p.proj_inp_comp_activation = 'mish'

    p.decoder_init_strategy = "forward_last_all"
    p.encoder = 'bilstm'
    p.use_cell_based_encoder = False
    # cell-based encoders: typed_rnn, ind_rnn, onlstm, lstm, gru, rnn; see models.base_s2s.base_seq2seq.py file
    # seq-based encoders: lstm, transformer, bilstm, aug_lstm, aug_bilstm; see models.base_s2s.stacked_encoder.py file
    p.cell_encoder_is_bidirectional = True     # any cell-based RNN encoder above could be bidirectional
    p.cell_encoder_uses_packed_sequence = False

    return p


if __name__ == '__main__':
    import sys
    from trialbot.utils.root_finder import find_root
    sys.path.insert(0, find_root('.SRC'))
    main()
