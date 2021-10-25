from trialbot.training import Registry
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.root_finder import find_root


@Registry.hparamset()
def cfq_mod_ent_tranx():
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 10  # in num of epochs
    p.OPTIM = "adabelief"
    p.batch_sz = 32

    p.emb_sz = 256
    p.src_namespace = 'questionPatternModEntities'
    p.tgt_namespace = 'modent_rule_seq'
    p.hidden_sz = 256
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "none"
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
def scholar_common():
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 150
    p.WEIGHT_DECAY = 0.
    p.OPTIM = "adabelief"
    p.ADAM_BETAS = (0.9, 0.999)
    p.batch_sz = 16

    p.hidden_sz = 300
    p.src_namespace = 'sent'
    p.tgt_namespace = 'rule_seq'
    p.dropout = .5
    p.decoder = "lstm"
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    p.lr_scheduler_kwargs = {"model_size": 400, "warmup_steps": 50} # noam lr_scheduler
    p.tied_decoder_embedding = False
    p.num_enc_layers = 1
    p.num_dec_layers = 1
    p.emb_sz = 100

    p.encoder = 'bilstm'
    p.enc_out_dim = 300
    p.enc_attn = "dot_product"
    p.dec_in_dim = p.enc_out_dim
    p.dec_out_dim = p.enc_out_dim

    p.enc_dec_trans_act = 'tanh'
    p.enc_dec_trans_usage = 'consistent'

    p.proj_in_dim = p.emb_sz

    p.enc_dropout = 0
    p.dec_dropout = 0.5
    p.dec_hist_attn = "none"
    p.dec_inp_composer = 'cat_mapping'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'cat_mapping'
    p.proj_inp_comp_activation = 'mish'
    p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"

    # p.cluster_iter_key = 'group_id'
    return p


@Registry.hparamset()
def atis_common():
    p = scholar_common()
    p.TRAINING_LIMIT = 100
    p.hidden_sz = 300
    p.lr_scheduler_kwargs = {"model_size": 600, "warmup_steps": 50} # noam lr_scheduler
    return p


@Registry.hparamset()
def advising_common():
    p = atis_common()
    return p


