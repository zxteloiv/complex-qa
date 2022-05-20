from trialbot.training import Registry
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.root_finder import find_root
import sys
sys.path.insert(0, find_root('.SRC'))


def main():
    from utils.trialbot.setup_cli import setup as setup_cli
    from libs2s import setup_common_bot
    import datasets.comp_gen_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    import datasets.cg_bundle_translator

    bot = setup_common_bot(setup_cli(translator='s2s', seed=2021))
    bot.run()


@Registry.hparamset()
def scholar_onlstm2seq():
    p = scholar()
    p.use_cell_based_encoder = True
    p.encoder = 'onlstm'
    p.enc_dec_trans_forced = True
    return p


@Registry.hparamset()
def scholar_seq2onlstm():
    p = scholar()
    p.use_cell_based_encoder = True
    p.decoder = 'onlstm'
    p.enc_dec_trans_forced = True
    return p


@Registry.hparamset()
def scholar_onlstm2onlstm():
    p = scholar()
    p.use_cell_based_encoder = True
    p.encoder = 'onlstm'
    p.decoder = 'onlstm'
    p.enc_dec_trans_forced = False
    return p


@Registry.hparamset()
def scholar():
    from libs2s import base_hparams
    p = base_hparams()
    p.TRAINING_LIMIT = 150
    p.WEIGHT_DECAY = 0.
    p.OPTIM = "adabelief"
    p.ADAM_BETAS = (0.9, 0.999)
    p.batch_sz = 16

    p.src_namespace = 'sent'
    p.tgt_namespace = 'sql'
    p.decoder_init_strategy = "forward_last_parallel"
    p.num_enc_layers = 3
    p.num_dec_layers = 3

    p.hidden_sz = 300
    p.emb_sz = 100
    p.enc_out_dim = p.hidden_sz
    p.dec_in_dim = p.hidden_sz
    p.dec_out_dim = p.hidden_sz
    p.proj_in_dim = p.emb_sz

    p.use_cell_based_encoder = False
    # cell-based encoders: typed_rnn, ind_rnn, onlstm, lstm, gru, rnn; see models.base_s2s.base_seq2seq.py file
    # seq-based encoders: lstm, transformer, bilstm, aug_lstm, aug_bilstm; see models.base_s2s.stacked_encoder.py file
    p.encoder = 'lstm'
    p.enc_attn = "dot_product"

    p.enc_dec_trans_usage = 'consistent'
    p.enc_dec_trans_act = 'tanh'
    p.enc_dec_trans_forced = False

    return p


@Registry.hparamset()
def atis():
    p = scholar()
    p.batch_sz = 32
    p.TRAINING_LIMIT = 80
    p.hidden_sz = 200
    p.encoder = 'lstm'
    return p


@Registry.hparamset()
def advising():
    p = atis()
    return p


@Registry.hparamset()
def geo():
    p = scholar()
    p.hidden_sz = 200
    return p


if __name__ == '__main__':
    main()
