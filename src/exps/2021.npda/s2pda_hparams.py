from typing import Dict
import sys
import os
import os.path as osp
from trialbot.training import TrialBot, Registry, Events
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.root_finder import find_root

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))
import datasets.cfq_translator
import datasets.cg_bundle_translator

@Registry.hparamset()
def cfq_pda():
    from trialbot.training.hparamset import HyperParamSet
    from trialbot.utils.root_finder import find_root
    ROOT = find_root()
    p = HyperParamSet.common_settings(ROOT)
    p.TRAINING_LIMIT = 10
    p.OPTIM = "adabelief"
    p.batch_sz = 32
    p.WEIGHT_DECAY = .1
    p.ADAM_BETAS = (0.9, 0.98)
    p.optim_kwargs = {"rectify": False}
    p.GRAD_CLIPPING = .2    # grad norm required to be <= 2

    p.src_ns = 'questionPatternModEntities'
    p.tgt_ns = datasets.cfq_translator.TREE_NS

    # transformer requires input embedding equal to hidden size
    p.encoder = "bilstm"
    p.num_enc_layers = 2
    p.enc_sz = 128
    p.emb_sz = 128
    p.hidden_sz = 128
    p.num_heads = 4

    p.enc_attn = 'generalized_bilinear'
    p.attn_use_linear = False
    p.attn_use_bias = False
    p.attn_use_tanh_activation = False

    p.dropout = .2
    p.num_expander_layer = 1
    p.expander_rnn = 'typed_rnn'    # typed_rnn, lstm
    p.max_derivation_step = 200
    p.grammar_entry = "queryunit"

    # ----------- tree encoder settings -------------

    p.tree_encoder = 're_zero_bilinear'

    p.bilinear_rank = 1
    p.bilinear_pool = p.hidden_sz // p.num_heads
    p.bilinear_linear = True
    p.bilinear_bias = True

    p.num_re0_layer = 6

    p.use_attn_residual_norm = True

    p.tree_self_attn = 'seq_mha'    # seq_mha, generalized_dot_product

    # ----------- end of tree settings -------------

    p.rule_scorer = "triple_inner_product" # heuristic, mlp, (triple|concat|add_inner_product)
    return p


@Registry.hparamset()
def sql_pda():
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 100
    p.OPTIM = "adabelief"
    p.batch_sz = 16
    p.WEIGHT_DECAY = 1e-6
    p.ADAM_LR = 1e-3
    p.ADAM_BETAS = (0.9, 0.999)
    p.optim_kwargs = {"rectify": True, "weight_decouple": True}
    p.GRAD_CLIPPING = 1    # grad norm required to be <= 2

    p.src_ns = 'sent'
    p.tgt_ns = datasets.cg_bundle_translator.TREE_NS

    # transformer requires input embedding equal to hidden size
    p.encoder = "lstm"
    p.num_enc_layers = 2
    p.overall_dim = 256
    p.enc_sz = p.overall_dim
    p.emb_sz = p.overall_dim
    p.hidden_sz = p.overall_dim
    p.num_heads = 4

    p.enc_attn = 'generalized_bilinear'
    p.attn_use_linear = False
    p.attn_use_bias = False
    p.attn_use_tanh_activation = False

    p.dropout = .2
    p.num_expander_layer = 1    # the expander is fixed to be lstm
    p.max_derivation_step = 200
    p.grammar_entry = "statement"

    # ----------- tree encoder settings -------------

    p.tree_encoder = 're_zero_bilinear'
    p.num_re0_layer = 6

    p.bilinear_rank = 1
    p.bilinear_pool = 8
    p.bilinear_linear = True
    p.bilinear_bias = True
    p.tree_self_attn = 'seq_mha'    # seq_mha, generalized_dot_product
    p.use_attn_residual_norm = True

    # ----------- end of tree settings -------------
    p.rule_scorer = "add_inner_product"  # heuristic, mlp, (triple|concat|add_inner_product)
    return p


