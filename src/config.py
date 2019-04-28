# config

import os.path
import json

# ======================
# general config

class HyperParamSet:
    def __str__(self):
        json_obj = dict((attr, getattr(self, attr)) for attr in dir(self)
                        if hasattr(self, attr) and not attr.startswith('_'))
        return json.dumps(json_obj)

def general_config():
    conf = HyperParamSet()

    conf.DEVICE = -1
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    conf.ROOT = ROOT
    conf.SNAPSHOT_PATH = os.path.join(ROOT, 'snapshots')

    conf.LOG_REPORT_INTERVAL = (1, 'iteration')
    conf.TRAINING_LIMIT = 500  # in num of epochs
    conf.SAVE_INTERVAL = (100, 'iteration')

    conf.ADAM_LR = 1e-3
    conf.ADAM_BETAS = (.9, .98)
    conf.ADAM_EPS = 1e-9

    conf.GRAD_CLIPPING = 5

    conf.SGD_LR = 1e-2
    conf.DATA_PATH = os.path.join(ROOT, 'data')

    return conf

# ======================
# dataset config

from utils.dataset_path import DatasetPath

DATA_PATH = general_config().DATA_PATH
DATASETS = dict()
DATASETS["atis"] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'atis', 'train.json'),
    dev_path=os.path.join(DATA_PATH, 'atis', 'dev.json'),
    test_path=os.path.join(DATA_PATH, 'atis', 'test.json'),
)
DATASETS["weibo"] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'weibo', 'train_data'),
    dev_path=os.path.join(DATA_PATH, 'weibo', 'valid_data'),
    test_path=os.path.join(DATA_PATH, 'weibo', 'test_data'),
)
DATASETS["small_weibo"] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'small_weibo', 'train_data'),
    dev_path=os.path.join(DATA_PATH, 'small_weibo', 'valid_data'),
    test_path=os.path.join(DATA_PATH, 'small_weibo', 'test_data'),
)
DATASETS['sqa'] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'sqa', 'train.tsv'),
    dev_path=os.path.join(DATA_PATH, 'sqa', 'dev.tsv'),
    test_path=os.path.join(DATA_PATH, 'sqa', 'test.tsv'),
)
DATASETS['geoqueries_sp'] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'geoqueries_sp', 'train.json'),
    dev_path=os.path.join(DATA_PATH, 'geoqueries_sp', 'dev.json'),
    test_path=os.path.join(DATA_PATH, 'geoqueries_sp', 'test.json'),
)

DATASETS['geoqueries'] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'geoqueries', 'orig.train.json'),
    dev_path="",
    test_path=os.path.join(DATA_PATH, 'geoqueries', 'test.json'),
)
DATASETS['wikisql'] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'wikisql', 'train.json'),
    dev_path="",
    test_path=os.path.join(DATA_PATH, 'wikisql', 'test.json'),
)
DATASETS['django'] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'django', 'train.json'),
    dev_path="",
    test_path=os.path.join(DATA_PATH, 'django', 'test.json'),
)

# ======================
# model config

def s2s_geoquery_conf():
    conf = general_config()
    conf.emb_sz = 50
    conf.batch_sz = 32
    conf.max_decoding_len = 50
    return conf

def s2s_atis_conf():
    conf = general_config()
    conf.emb_sz = 200
    conf.batch_sz = 32
    conf.max_decoding_len = 60
    return conf

def t2s_geoquery_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 32
    conf.max_decoding_len = 50
    return conf

def t2s_atis_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 32
    conf.max_decoding_len = 50
    conf.num_heads = 8
    conf.max_num_layers = 1
    conf.act = False
    conf.residual_dropout = .1
    conf.attention_dropout = .1
    conf.feedforward_dropout = .1
    conf.vanilla_wiring = False
    return conf

def base_s2s_atis_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 20
    conf.max_decoding_len = 60
    conf.num_heads = 8
    conf.num_enc_layers = 2
    conf.encoder = 'lstm'
    conf.decoder = 'lstm'
    conf.residual_dropout = .1
    conf.attention_dropout = .1
    conf.feedforward_dropout = .1
    conf.intermediate_dropout = .5
    conf.vanilla_wiring = False
    conf.enc_attn = "dot_product"
    conf.dec_hist_attn = "dot_product"
    conf.dec_cell_height = 2
    conf.concat_attn_to_dec_input = True
    return conf

def unc_s2s_geoquery_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 20
    conf.max_decoding_len = 60
    conf.num_heads = 2
    conf.num_enc_layers = 2
    conf.encoder = 'bilstm'
    conf.residual_dropout = .1
    conf.attention_dropout = .1
    conf.feedforward_dropout = .1
    conf.intermediate_dropout = .5
    conf.vanilla_wiring = True
    conf.decoder = 'n_lstm'
    conf.enc_attn = "bilinear"
    conf.dec_hist_attn = "dot_product"
    conf.dec_cell_height = 2
    conf.concat_attn_to_dec_input = True
    conf.model_mode = 0,   # 0: train s2s; 1: train RL unc; 2: joint
    conf.scheduled_sampling = .2
    conf.pondering_limit = 3
    conf.uncertainty_sample_num = 5
    conf.uncertainty_loss_weight = 1.
    conf.reward_discount = .5
    return conf

def unc_s2s_atis_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 20
    conf.max_decoding_len = 60
    conf.num_heads = 2
    conf.num_enc_layers = 2
    conf.encoder = 'lstm'
    conf.residual_dropout = .1
    conf.attention_dropout = .1
    conf.feedforward_dropout = .1
    conf.intermediate_dropout = .5
    conf.vanilla_wiring = True
    conf.decoder = 'n_lstm'
    conf.enc_attn = "bilinear"
    conf.dec_hist_attn = "dot_product"
    conf.dec_cell_height = 2
    conf.concat_attn_to_dec_input = True
    conf.model_mode = 0,   # 0: train s2s; 1: train RL unc; 2: joint
    conf.scheduled_sampling = .2
    conf.pondering_limit = 3
    conf.uncertainty_sample_num = 5
    conf.uncertainty_loss_weight = 1.
    conf.reward_discount = .5
    return conf

def ada_t2s_atis_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 20
    conf.max_decoding_len = 60
    conf.num_heads = 8
    conf.num_enc_layers = 2
    conf.encoder = 'lstm'
    conf.decoder = 'lstm'
    conf.act_max_layer = 3
    conf.act = False
    conf.act_dropout = .3
    conf.act_epsilon = .1
    conf.residual_dropout = .1
    conf.attention_dropout = .1
    conf.feedforward_dropout = .1
    conf.embedding_dropout = .5
    conf.decoder_dropout = .5
    conf.prediction_dropout = .4
    conf.vanilla_wiring = False
    conf.act_loss_weight = -0.1
    conf.dwa = "dot_product"
    conf.enc_attn = "dot_product"
    conf.dec_hist_attn = "dot_product"
    conf.act_mode = 'mean_field'
    conf.dec_cell_height = 2
    return conf

def transformer_weibo_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 128
    conf.max_decoding_len = 15
    conf.num_heads = 8
    conf.num_layers = 6
    conf.feedforward_dropout = 0.1
    return conf

def transformer_geoquery_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 32
    conf.max_decoding_len = 60
    conf.num_heads = 8
    conf.num_layers = 6
    conf.feedforward_dropout = 0.1
    return conf

def transformer_atis_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 32
    conf.max_decoding_len = 70
    conf.num_heads = 8
    conf.num_layers = 2
    conf.feedforward_dropout = 0.1
    return conf

def ut_geoquery_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 100
    conf.max_decoding_len = 60
    conf.num_heads = 8
    conf.max_num_layers = 1
    conf.act = True
    return conf

def ut_atis_conf():
    conf = general_config()
    conf.emb_sz = 256
    conf.batch_sz = 100
    conf.max_decoding_len = 70
    conf.num_heads = 8
    conf.max_num_layers = 1
    conf.act = True
    conf.residual_dropout = .05
    conf.attention_dropout = .001
    conf.feedforward_dropout = .05
    conf.vanilla_wiring = False
    return conf

SETTINGS = [x for x in dir() if x.endswith('_conf')]

