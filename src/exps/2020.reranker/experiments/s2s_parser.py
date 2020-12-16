from os.path import join, abspath, dirname
import sys
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..', '..')))   # up to src
from trialbot.training import TrialBot, Events, Registry

def main():
    from utils.trialbot_setup import setup
    from models.base_s2s.base_seq2seq import BaseSeq2Seq
    args = setup(seed=2020)
    bot = TrialBot(trial_name='s2s_parser', get_model_func=BaseSeq2Seq.from_param_and_vocab, args=args)

@Registry.hparamset()
def s2s_top():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    p = HyperParamSet.common_settings(find_root())
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

if __name__ == '__main__':
    main()