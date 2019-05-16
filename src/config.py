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

def common_settings():
    hparams = HyperParamSet()

    hparams.DEVICE = -1
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    hparams.ROOT = ROOT
    hparams.SNAPSHOT_PATH = os.path.join(ROOT, 'snapshots')

    hparams.LOG_REPORT_INTERVAL = (1, 'iteration')
    hparams.TRAINING_LIMIT = 500  # in num of epochs
    hparams.SAVE_INTERVAL = (100, 'iteration')
    hparams.batch_sz = 128
    hparams.min_word = 3

    hparams.ADAM_LR = 1e-3
    hparams.ADAM_BETAS = (.9, .98)
    hparams.ADAM_EPS = 1e-9

    hparams.GRAD_CLIPPING = 5

    hparams.SGD_LR = 1e-2
    hparams.DATA_PATH = os.path.join(ROOT, 'data')

    return hparams

# ======================
# dataset config

from utils.dataset_path import DatasetPath

DATA_PATH = common_settings().DATA_PATH
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
DATASETS['spider'] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'spider', 'train_spider.json'),
    dev_path=os.path.join(DATA_PATH, 'spider', 'dev.json'),
    test_path=os.path.join(DATA_PATH, 'spider', 'dev.json'),
)
DATASETS['dialogue'] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'dialogue_data', 'dialogue_train_data'),
    dev_path=os.path.join(DATA_PATH, 'dialogue_data', 'dialogue_valid_data'),
    test_path=os.path.join(DATA_PATH, 'dialogue_data', 'dialogue_test_data'),
)
DATASETS['mspars'] = DatasetPath(
    train_path=os.path.join(DATA_PATH, 'mspars', 'MSParS.train'),
    dev_path=os.path.join(DATA_PATH, 'mspars', 'MSParS.dev'),
    test_path=os.path.join(DATA_PATH, 'mspars', 'MSParS.dev'),
)

# ======================
# model config

def base_s2s_atis_hparams():
    hparams = common_settings()
    hparams.emb_sz = 256
    hparams.batch_sz = 20
    hparams.max_decoding_len = 60
    hparams.num_heads = 8
    hparams.num_enc_layers = 2
    hparams.encoder = 'lstm'
    hparams.decoder = 'lstm'
    hparams.residual_dropout = .1
    hparams.attention_dropout = .1
    hparams.feedforward_dropout = .1
    hparams.intermediate_dropout = .5
    hparams.vanilla_wiring = False
    hparams.enc_attn = "dot_product"
    hparams.dec_hist_attn = "dot_product"
    hparams.dec_cell_height = 2
    hparams.concat_attn_to_dec_input = True
    return hparams

def unc_s2s_geoquery_hparams():
    hparams = common_settings()
    hparams.emb_sz = 256
    hparams.batch_sz = 20
    hparams.max_decoding_len = 60
    hparams.num_heads = 2
    hparams.num_enc_layers = 2
    hparams.encoder = 'bilstm'
    hparams.residual_dropout = .1
    hparams.attention_dropout = .1
    hparams.feedforward_dropout = .1
    hparams.intermediate_dropout = .5
    hparams.vanilla_wiring = True
    hparams.decoder = 'n_lstm'
    hparams.enc_attn = "bilinear"
    hparams.dec_hist_attn = "dot_product"
    hparams.dec_cell_height = 2
    hparams.concat_attn_to_dec_input = True
    hparams.model_mode = 0,   # 0: train s2s; 1: train RL unc; 2: joint
    hparams.scheduled_sampling = .2
    hparams.pondering_limit = 3
    hparams.uncertainty_sample_num = 5
    hparams.uncertainty_loss_weight = 1.
    hparams.reward_discount = .5
    return hparams

def unc_s2s_atis_hparams():
    hparams = common_settings()
    hparams.emb_sz = 256
    hparams.batch_sz = 20
    hparams.max_decoding_len = 60
    hparams.num_heads = 2
    hparams.num_enc_layers = 2
    hparams.encoder = 'lstm'
    hparams.residual_dropout = .1
    hparams.attention_dropout = .1
    hparams.feedforward_dropout = .1
    hparams.intermediate_dropout = .5
    hparams.vanilla_wiring = True
    hparams.decoder = 'n_lstm'
    hparams.enc_attn = "bilinear"
    hparams.dec_hist_attn = "dot_product"
    hparams.dec_cell_height = 2
    hparams.concat_attn_to_dec_input = True
    hparams.model_mode = 0,   # 0: train s2s; 1: train RL unc; 2: joint
    hparams.scheduled_sampling = .2
    hparams.pondering_limit = 3
    hparams.uncertainty_sample_num = 5
    hparams.uncertainty_loss_weight = 1.
    hparams.reward_discount = .5
    return hparams

def ada_t2s_atis_hparams():
    hparams = common_settings()
    hparams.emb_sz = 256
    hparams.batch_sz = 20
    hparams.max_decoding_len = 60
    hparams.num_heads = 8
    hparams.num_enc_layers = 2
    hparams.encoder = 'lstm'
    hparams.decoder = 'lstm'
    hparams.act_max_layer = 3
    hparams.act = False
    hparams.act_dropout = .3
    hparams.act_epsilon = .1
    hparams.residual_dropout = .1
    hparams.attention_dropout = .1
    hparams.feedforward_dropout = .1
    hparams.embedding_dropout = .5
    hparams.decoder_dropout = .5
    hparams.prediction_dropout = .4
    hparams.vanilla_wiring = False
    hparams.act_loss_weight = -0.1
    hparams.dwa = "dot_product"
    hparams.enc_attn = "dot_product"
    hparams.dec_hist_attn = "dot_product"
    hparams.act_mode = 'mean_field'
    hparams.dec_cell_height = 2
    return hparams

def transformer_dialogue_hparams():
    hparams = common_settings()
    hparams.emb_sz = 256
    hparams.batch_sz = 128
    hparams.max_decoding_len = 20
    hparams.num_heads = 8
    hparams.num_layers = 3
    hparams.feedforward_dropout = 0.1
    hparams.ADAM_LR = 1e-4
    return hparams

def transformer_atis_hparams():
    hparams = common_settings()
    hparams.emb_sz = 256
    hparams.batch_sz = 32
    hparams.max_decoding_len = 70
    hparams.num_heads = 8
    hparams.num_layers = 2
    hparams.feedforward_dropout = 0.1
    return hparams

SETTINGS = dict((k, v) for k, v in globals().items() if k.endswith('_hparams'))

def register_hparams(func):
    global SETTINGS
    SETTINGS[func.__name__] = func
    return func
