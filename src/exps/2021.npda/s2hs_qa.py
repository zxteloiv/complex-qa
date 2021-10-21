# sequence to hierarchical sequence model applied to SP-based QA
import sys, os.path as osp
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))
from trialbot.training import TrialBot
from trialbot.training import Registry

from models.base_s2s.seq2hseq import Seq2HierSeq
import datasets.comp_gen_bundle as cg_bundle
import datasets.cg_bundle_translator
cg_bundle.install_parsed_qa_datasets(Registry._datasets)


@Registry.hparamset()
def crude_conf():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    ROOT = find_root()
    p = HyperParamSet.common_settings(ROOT)
    p.TRAINING_LIMIT = 50
    p.batch_sz = 32
    p.WEIGHT_DECAY = .2
    p.OPTIM = "AdaBelief"

    p.emb_sz = 256
    p.src_namespace = 'sent'
    p.tgt_namespace = 'formal_token'
    p.hidden_sz = 128

    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.enc_out_dim = p.hidden_sz

    p.dec_in_dim = p.hidden_sz  # by default
    p.dec_out_dim = p.hidden_sz  # by default
    p.proj_in_dim = p.hidden_sz  # by default
    p.enc_attn = 'bilinear'
    p.word_enc_attn = "none"        # disabled
    p.char_enc_attn = p.enc_attn    # by default
    p.word_dec_inp_composer = 'none'
    p.char_dec_inp_composer = 'cat_mapping'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'cat_mapping'
    p.proj_inp_comp_activation = 'mish'

    p.dropout = .2
    p.enc_dropout = p.dropout  # by default
    p.dec_dropout = p.dropout  # by default

    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.max_word_step = 50
    p.max_char_step = 10

    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    p.tied_decoder_embedding = False
    # p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"
    return p


def main():
    from utils.trialbot.setup import setup
    from trialbot.training import Events
    from trialbot.training.extensions import every_epoch_model_saver
    from utils.trialbot.extensions import print_hyperparameters, get_metrics, print_models
    from utils.trialbot.extensions import evaluation_on_dev_every_epoch
    from utils.trialbot.extensions import end_with_nan_loss
    from utils.trialbot.extensions import collect_garbage
    args = setup(seed=2021, hparamset='crude_conf', translator='gd')
    bot = TrialBot(trial_name='s2hs', get_model_func=Seq2HierSeq.from_param_and_vocab, args=args)
    bot.add_event_handler(Events.STARTED, print_models, 100)
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 90)
    bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
    bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 95)
    # bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100)
    bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
    if not args.test:
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90,
                              rewrite_eval_hparams={"batch_sz": 32})
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 80,
                              rewrite_eval_hparams={"batch_sz": 32}, on_test_data=True)

    bot.run()


if __name__ == '__main__':
    main()