# mostly a seq2seq but the encoder is modified for the Diora or S-Diora model

from trialbot.training import Registry
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.root_finder import find_root
import sys
sys.path.insert(0, find_root('.SRC'))


def main():
    from utils.trialbot.setup import setup
    from libs2s import run_exp
    from models.diora.diora2seq import Diora2Seq
    import datasets.comp_gen_bundle as cg_bundle
    cg_bundle.install_parsed_qa_datasets(Registry._datasets)
    import datasets.cg_bundle_translator

    bot = run_exp(setup(translator='tranx', seed=2021, device=0),
                  get_model_func=Diora2Seq.from_param_and_vocab)
    bot.run()


@Registry.hparamset()
def sch_tranx():
    p = _base_hparams()
    # only list some crucial changes here
    p.TRAINING_LIMIT = 150
    p.encoder = 'diora'
    p.enc_attn = "none"
    p.dec_hist_attn = "none"
    p.dec_inp_composer = 'none'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'none'
    p.proj_inp_comp_activation = 'mish'
    p.diora_topk = 1
    return p

@Registry.hparamset()
def sch_s2s():
    p = _base_hparams()
    p.tgt_namespace = 'sql'
    p.encoder = 'diora'
    p.enc_attn = "none"
    p.dec_hist_attn = "none"
    p.dec_inp_composer = 'none'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'none'
    p.proj_inp_comp_activation = 'mish'
    p.diora_topk = 1
    return p


def _base_hparams():
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 150
    p.WEIGHT_DECAY = 0.
    p.OPTIM = "adabelief"
    p.ADAM_BETAS = (0.9, 0.999)
    p.batch_sz = 16

    p.lr_scheduler_kwargs = {'model_size': 400, 'warmup_steps': 50}
    p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"

    p.hidden_sz = 300
    p.src_namespace = 'sent'
    p.tgt_namespace = 'rule_seq'
    p.dropout = .5
    p.decoder = "lstm"
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.num_dec_layers = 1
    p.tied_decoder_embedding = False
    p.emb_sz = 100

    p.use_cell_based_encoder = False
    # cell-based encoders: typed_rnn, ind_rnn, onlstm, lstm, gru, rnn; see models.base_s2s.base_seq2seq.py file
    # seq-based encoders: lstm, transformer, bilstm, aug_lstm, aug_bilstm; see models.base_s2s.stacked_encoder.py file
    p.cell_encoder_is_bidirectional = True     # any cell-based RNN encoder above could be bidirectional
    p.cell_encoder_uses_packed_sequence = False

    p.encoder = 's-diora'
    p.diora_topk = 2
    p.enc_out_dim = p.hidden_sz
    p.dec_in_dim = p.hidden_sz
    p.dec_out_dim = p.hidden_sz

    p.enc_dec_trans_usage = 'dec_init'
    p.enc_dec_trans_act = 'tanh'
    p.enc_dec_trans_forced = True

    p.proj_in_dim = p.emb_sz

    p.enc_dropout = 0
    p.dec_dropout = 0.5
    p.enc_attn = "none"
    p.dec_hist_attn = "none"
    p.dec_inp_composer = 'none'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'none'
    p.proj_inp_comp_activation = 'mish'

    # ============ backward compatibility ==============
    # this attr must be set but is not useful in diora because any diora has a fixed strategy
    p.decoder_init_strategy = "forward_last_all"
    p.num_enc_layers = 1

    return p

if __name__ == '__main__':
    main()

