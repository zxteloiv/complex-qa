from typing import Dict
import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import logging
from collections import defaultdict

from trialbot.training import TrialBot, Registry
from trialbot.data import NSVocabulary
from trialbot.training.updater import TrainingUpdater

import datasets.cfq
import datasets.cfq_translator

@Registry.hparamset()
def cfq_pda():
    from trialbot.training.hparamset import HyperParamSet
    from utils.root_finder import find_root
    ROOT = find_root()
    p = HyperParamSet.common_settings(ROOT)
    p.TRAINING_LIMIT = 10
    p.OPTIM = "RAdam"
    p.batch_sz = 32
    p.weight_decay = .2

    p.tutor_usage = "from_dataset" # from_grammar, from_dataset

    p.src_ns = 'questionPatternModEntities'
    p.tgt_ns = datasets.cfq_translator.UNIFIED_TREE_NS

    # transformer requires input embedding equal to hidden size
    p.encoder = "lstm"
    p.num_enc_layers = 2
    p.emb_sz = 128
    p.hidden_sz = 128
    p.num_heads = 4

    p.enc_attn = "generalized_dot_product"
    p.attn_use_linear = False
    p.attn_use_bias = False
    p.attn_use_tanh_activation = False

    p.dropout = .4
    p.num_expander_layer = 1
    p.max_derivation_step = 200
    p.max_expansion_len = 11
    p.grammar_entry = "queryunit"

    p.tree_encoder = 'lstm' # lstm, bare_dot_prod_attn, bilinear_tree_lstm, re_zero_bilinear
    p.use_attn_residual_norm = True
    p.bare_attn_activation = 'linear'  # tanh, linear
    p.bare_attn_pre_activation = 'tanh'  # tanh, linear
    p.bilinear_rank = 1
    p.bilinear_pool = p.hidden_sz // p.num_heads
    p.bilinear_linear = False   # by default do not use linear mappings within bilinear modules
    p.bilinear_bias = False     # by default do not use bias within bilinear modules
    p.num_re_zero_layer = 18
    p.detach_tree_encoder = False   # whether to stop-gradient for the tree encoder parameters
    p.tree_training_lr_factor = 1.  # applied on the learning rate for tree encoder parameters

    p.exact_token_predictor = "quant" # linear, mos, quant
    p.num_exact_token_mixture = 1
    p.exact_token_quant_criterion = "dot_product"

    p.rule_scorer = "triple_inner_product" # heuristic, mlp, triple_inner_product, triple_cosine
    return p

@Registry.hparamset()
def cfq_pda_0():
    p = cfq_pda()
    p.emb_sz = p.hidden_sz = 256
    return p

@Registry.hparamset()
def cfq_pda_1():
    p = cfq_pda()
    p.tree_training_lr_factor = 1e-2
    return p

@Registry.hparamset()
def cfq_pda_2():
    p = cfq_pda()
    p.emb_sz = p.hidden_sz = 256
    p.tree_training_lr_factor = 1e-2
    return p

@Registry.hparamset()
def cfq_pda_3():
    p = cfq_pda()
    p.emb_sz = p.hidden_sz = 256
    p.tree_encoder = 're_zero_bilinear'
    p.tree_training_lr_factor = 1e-2
    return p

def get_grammar_tutor(p, vocab):
    ns_symbol, ns_exact_token = p.tgt_ns
    import torch
    from models.neural_pda.grammar_tutor import GrammarTutor
    from datasets.lark_translator import UnifiedLarkTranslator
    grammar_dataset, _, _ = datasets.cfq.sparql_pattern_grammar()
    grammar_translator = UnifiedLarkTranslator(datasets.cfq_translator.UNIFIED_TREE_NS)
    grammar_translator.index_with_vocab(vocab)
    rule_list = [grammar_translator.to_tensor(r) for r in grammar_dataset]
    rule_batch = grammar_translator.batch_tensor(rule_list)
    ordered_lhs, ordered_rhs = list(zip(*rule_batch.items()))
    ordered_rhs = torch.stack(ordered_rhs)
    gt = GrammarTutor(vocab.get_vocab_size(ns_symbol), ordered_lhs, ordered_rhs)

    from datasets.lark_translator import LarkExactTokenReader
    from models.neural_pda.token_tutor import ExactTokenTutor
    reader = LarkExactTokenReader(vocab=vocab, ns=(ns_symbol, ns_exact_token))
    grammar_dataset, _, _ = datasets.cfq.sparql_pattern_grammar()
    pair_set = set()
    for r in grammar_dataset:
        for p in reader.generate_symbol_token_pair(r):
            pair_set.add(p)
    valid_token_lookup = reader.merge_the_pairs(pair_set)
    ett = ExactTokenTutor(vocab.get_vocab_size(ns_symbol), vocab.get_vocab_size(ns_exact_token), valid_token_lookup)

    return ett, gt

def get_cfq_tailored_tutor(p, vocab):
    from datasets.cfq_translator import CFQTutorBuilder
    from tqdm import tqdm
    import pickle
    from models.neural_pda.token_tutor import ExactTokenTutor
    from models.neural_pda.grammar_tutor import GrammarTutor
    from utils.preprocessing import nested_list_numbers_to_tensors

    tutor_dump_name = os.path.join(datasets.cfq.CFQ_PATH, 'tutors.pkl')
    if os.path.exists(tutor_dump_name):
        logging.getLogger(__name__).info(f"Use the existing tutor dump... {tutor_dump_name}")
        tutor_repr = pickle.load(open(tutor_dump_name, 'rb'))
    else:
        logging.getLogger(__name__).info(f"Build the tutor dump for the first time...")

        builder = CFQTutorBuilder()
        builder.index_with_vocab(vocab)
        train_set, _, _ = datasets.cfq.cfq_mcd1()
        tutor_repr = builder.batch_tensor([builder.to_tensor(example) for example in tqdm(train_set)])
        logging.getLogger(__name__).info(f"Dump the tutor information... {tutor_dump_name}")
        pickle.dump(tutor_repr, open(tutor_dump_name, 'wb'))

    token_map: defaultdict
    grammar_map: defaultdict
    token_map, grammar_map = map(tutor_repr.get, ('token_map', 'grammar_map'))
    ns_s, ns_et = p.tgt_ns
    ett = ExactTokenTutor(vocab.get_vocab_size(ns_s), vocab.get_vocab_size(ns_et), token_map, False)

    ordered_lhs, ordered_rhs_options = list(zip(*grammar_map.items()))
    ordered_rhs_options = nested_list_numbers_to_tensors(ordered_rhs_options)
    gt = GrammarTutor(vocab.get_vocab_size(ns_s), ordered_lhs, ordered_rhs_options)
    return ett, gt

def get_tree_encoder(p, vocab):
    import models.neural_pda.partial_tree_encoder as partial_tree
    from models.modules.decomposed_bilinear import DecomposedBilinear
    from allennlp.nn.activations import Activation

    # embedding will be transformed into hid size with lhs_symbol_mapper,
    # thus tree encoder input will be hid_sz by default
    if p.tree_encoder == 'lstm':
        tree_encoder = partial_tree.TopDownLSTMEncoder(
            p.hidden_sz, p.hidden_sz, p.hidden_sz // 2, dropout=p.dropout
        )

    elif p.tree_encoder == 'bare_dot_prod_attn':
        tree_encoder = partial_tree.BareDotProdAttnEncoder(
            activation=Activation.by_name(p.bare_attn_activation)(),
            pre_activation=Activation.by_name(p.bare_attn_pre_activation)(),
        )

    elif p.tree_encoder == 'bilinear_tree_lstm':
        tree_encoder = partial_tree.TopDownBilinearLSTMEncoder(
            p.hidden_sz, p.hidden_sz,
            p.bilinear_rank, p.bilinear_pool, p.bilinear_linear, p.bilinear_bias, p.dropout,
        )

    elif p.tree_encoder.startswith('re_zero'):
        if p.tree_encoder.endswith('bilinear'):
            layer_encoder = partial_tree.SingleStepBilinear(DecomposedBilinear(
                p.hidden_sz, p.hidden_sz, p.hidden_sz,
                p.bilinear_rank, p.bilinear_pool, p.bilinear_linear, p.bilinear_bias, p.dropout,
            ))
        elif p.tree_encoder.endswith('dot_prod'):
            layer_encoder = partial_tree.SingleStepDotProd()
        else:
            raise NotImplementedError

        tree_encoder = partial_tree.ReZeroEncoder(num_layers=p.num_re_zero_layer, layer_encoder=layer_encoder)

    else:
        raise NotImplementedError

    return tree_encoder

def get_model(p, vocab: NSVocabulary):
    from torch import nn
    from models.neural_pda.seq2pda import Seq2PDA
    from models.neural_pda.npda import NeuralPDA
    from models.transformer.multi_head_attention import MultiHeadSelfAttention
    from models.neural_pda.rule_scorer import MLPScorerWrapper, HeuristicMLPScorerWrapper, GeneralizedInnerProductScorer
    from models.modules.stacked_encoder import StackedEncoder
    from models.modules.attention_wrapper import get_wrapped_attention
    from models.modules.quantized_token_predictor import QuantTokenPredictor
    from models.modules.stacked_rnn_cell import StackedRNNCell, StackedLSTMCell, RNNType
    from models.modules.sym_typed_rnn_cell import SymTypedRNNCell
    from models.modules.container import MultiInputsSequential, UnpackedInputsSequential, SelectArgsById
    from models.modules.mixture_softmax import MoSProjection
    from allennlp.nn.activations import Activation

    encoder = StackedEncoder.get_encoder(p)
    emb_src = nn.Embedding(vocab.get_vocab_size(p.src_ns), embedding_dim=p.emb_sz)

    ns_s, ns_et = p.tgt_ns
    emb_s = nn.Embedding(vocab.get_vocab_size(ns_s), p.emb_sz)

    if p.exact_token_predictor == "linear":
        exact_token_predictor = nn.Sequential(
            nn.Linear(encoder.get_output_dim() + p.hidden_sz + p.emb_sz, vocab.get_vocab_size(ns_et)),
        )
    elif p.exact_token_predictor == "quant":
        exact_token_predictor = QuantTokenPredictor(
            vocab.get_vocab_size(ns_et), encoder.get_output_dim() + p.hidden_sz + p.emb_sz,
            quant_criterion=p.exact_token_quant_criterion,
        )
    else:
        exact_token_predictor = MoSProjection(
            p.num_exact_token_mixture, encoder.get_output_dim() + p.hidden_sz + p.emb_sz, vocab.get_vocab_size(ns_et),
        )

    if p.tutor_usage == "from_grammar":
        ett, gt = get_grammar_tutor(p, vocab)
    else:
        ett, gt = get_cfq_tailored_tutor(p, vocab)

    if p.rule_scorer == "heuristic":
        assert encoder.get_output_dim() == p.hidden_sz, "attention outputs must have the same size with the hidden_size"
        rule_scorer = HeuristicMLPScorerWrapper(MultiInputsSequential(
            nn.Linear(encoder.get_output_dim() + p.hidden_sz * 2 + p.hidden_sz * 2, p.hidden_sz),
            nn.Dropout(p.dropout),
            Activation.by_name('tanh')(),
            nn.Linear(p.hidden_sz, 1),
        ))
    elif p.rule_scorer == "triple_inner_product":
        rule_scorer = GeneralizedInnerProductScorer()
    elif p.rule_scorer == "triple_cosine":
        rule_scorer = GeneralizedInnerProductScorer(normalized=True)
    else:
        rule_scorer = MLPScorerWrapper(MultiInputsSequential(
            nn.Linear(encoder.get_output_dim() + p.hidden_sz * 2, p.hidden_sz),
            nn.Dropout(p.dropout),
            Activation.by_name('tanh')(),
            nn.Linear(p.hidden_sz, 1),
        ))

    # although the tree encoder construction needs not be here,
    # to keep the initialization exactly the same numerically as before (when the best baseline is achieved),
    # move the construction elsewhere will cause the model initialization different.
    # Though a robust algorithm should not be so sensitive, we just keep things untouched for better comparison.
    tree_encoder = get_tree_encoder(p, vocab)

    npda = NeuralPDA(
        symbol_embedding=emb_s,
        lhs_symbol_mapper=MultiInputsSequential(
            nn.Linear(p.emb_sz, p.hidden_sz // p.num_heads),
            nn.Dropout(p.dropout),
            nn.Linear(p.hidden_sz // p.num_heads, p.hidden_sz),
            nn.Tanh(),
        ) if p.emb_sz != p.hidden_sz else Activation.by_name('linear')(),   # `linear` returns the identity function
        grammar_tutor=gt,
        rhs_expander=StackedRNNCell(
            [
                SymTypedRNNCell(input_dim=p.emb_sz + 3 if floor == 0 else p.hidden_sz,
                                output_dim=p.hidden_sz,
                                nonlinearity="tanh")
                for floor in range(p.num_expander_layer)
            ],
            p.emb_sz, p.hidden_sz, p.num_expander_layer, intermediate_dropout=p.dropout
        ),
        rule_scorer=rule_scorer,
        exact_token_predictor=exact_token_predictor,
        token_tutor=ett,

        pre_tree_encoder=tree_encoder, #get_tree_encoder(p, vocab),
        pre_tree_self_attn=UnpackedInputsSequential(
            MultiHeadSelfAttention(p.num_heads, p.hidden_sz, p.hidden_sz, p.hidden_sz, 0.,),
            SelectArgsById(0),
        ),
        residual_norm_after_self_attn=nn.LayerNorm(p.hidden_sz) if p.use_attn_residual_norm else None,

        grammar_entry=vocab.get_token_index(p.grammar_entry, ns_s),
        max_derivation_step=p.max_derivation_step,
        dropout=p.dropout,
        detach_tree_encoder=p.detach_tree_encoder,
    )

    enc_attn_net = get_wrapped_attention(p.enc_attn, p.hidden_sz, encoder.get_output_dim(),
                                         num_heads=p.num_heads,
                                         use_linear=p.attn_use_linear,
                                         use_bias=p.attn_use_bias,
                                         use_tanh_activation=p.attn_use_tanh_activation,
                                         )


    model = Seq2PDA(
        encoder=encoder,
        src_embedding=emb_src,
        enc_attn_net=enc_attn_net,
        npda=npda,
        src_ns=p.src_ns,
        tgt_ns=p.tgt_ns,
        vocab=vocab,
    )

    return model

class PDATrainingUpdater(TrainingUpdater):
    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'PDATrainingUpdater':
        import torch
        from trialbot.data import RandomIterator
        from models.neural_pda.seq2pda import Seq2PDA
        model: Seq2PDA
        args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger

        if p.tree_training_lr_factor < 1:
            lr_conds = [
                (lambda k: 'pre_tree_encoder' in k, p.tree_training_lr_factor),
                (lambda k: 'npda' in k and 'embedder' in k, 1 - 0.2 * p.tree_training_lr_factor),
            ]
            params = [
                { 'params': list(v for k, v in model.named_parameters() if cond(k)), 'lr': p.ADAM_LR * lr_factor }
                for cond, lr_factor in lr_conds
            ]
            others = list(v for k, v in model.named_parameters() if all(not cond(k) for cond, _ in lr_conds))
            if len(others) > 0:
                params.append({'params': others})
            logger.info(f"parameters group defined as {params}")

        else:
            params = model.parameters()

        if hasattr(p, "OPTIM") and isinstance(p.OPTIM, str) and p.OPTIM.lower() == "sgd":
            optim = torch.optim.SGD(params, p.SGD_LR, weight_decay=p.WEIGHT_DECAY)
        elif hasattr(p, "OPTIM") and isinstance(p.OPTIM, str) and p.OPTIM.lower() == "radam":
            from radam import RAdam
            optim = RAdam(params, lr=p.ADAM_LR, weight_decay=p.WEIGHT_DECAY)
        else:
            optim = torch.optim.Adam(params, p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY)

        logger.info(f"Using the optimizer {str(optim)}")

        repeat_iter = shuffle_iter = not args.debug
        iterator = RandomIterator(bot.train_set, p.batch_sz, bot.translator, shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        updater = cls(model, iterator, optim, args.device, args.dry_run)
        return updater

def main():
    from utils.trialbot_setup import setup
    args = setup(seed=2021)

    bot = TrialBot(trial_name="s2pda_qa", get_model_func=get_model, args=args)

    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def get_metric(bot: TrialBot):
        import json
        print(json.dumps(bot.model.get_metric(reset=True, diagnosis=True)))

    @bot.attach_extension(Events.STARTED)
    def print_models(bot: TrialBot):
        print(str(bot.models))

    from utils.trial_bot_extensions import collect_garbage, print_hyperparameters
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 100)
    from trialbot.training import Events
    if not args.test:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import end_with_nan_loss
        from utils.trial_bot_extensions import evaluation_on_dev_every_epoch
        from utils.trial_bot_extensions import save_model_every_num_iters
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90, skip_first_epochs=1)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 80)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 80)

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def iteration_metric(bot: TrialBot):
            import json
            print(json.dumps(bot.model.get_metric()))

        # from utils.trial_bot_extensions import init_tensorboard_writer
        # from utils.trial_bot_extensions import write_batch_info_to_tensorboard
        # from utils.trial_bot_extensions import close_tensorboard
        # bot.add_event_handler(Events.STARTED, init_tensorboard_writer, 100)
        # bot.add_event_handler(Events.ITERATION_COMPLETED, write_batch_info_to_tensorboard, 100)
        # bot.add_event_handler(Events.COMPLETED, close_tensorboard, 100)

        # debug strange errors by inspecting running time, data size, etc.
        if args.debug:
            from utils.trial_bot_extensions import track_pytorch_module_forward_time
            bot.add_event_handler(Events.STARTED, track_pytorch_module_forward_time, 100)

        bot.updater = PDATrainingUpdater.from_bot(bot)

    else:
        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def iteration_metric(bot: TrialBot):
            import json
            print(json.dumps(bot.model.get_metric()))

        if args.debug:
            bot.add_event_handler(Events.ITERATION_COMPLETED, prediction_analysis, 100)
    bot.run()

def prediction_analysis(bot: TrialBot):
    output = bot.state.output
    if output is None:
        return

    output = bot.model.make_human_readable_output(output)
    batch_src = output['source_surface']
    batch_pred = output['prediction_surface']
    batch_gold = output['target_surface']
    batch_symbol = output['symbol_surface']
    batch_gold_symbol = output['rhs_symbol_surface']

    for src, pred, gold, p_s, g_s in zip(batch_src, batch_pred, batch_gold, batch_symbol, batch_gold_symbol):
        print('---' * 30)
        print("SRC:  " + " ".join(src))
        print("PRED: " + " ".join(pred))
        print("GOLD: " + " ".join(gold))
        print("PRED_symbol: " + " ".join(p_s))
        print("GOLD_symbol: " + " ".join(g_s))

if __name__ == '__main__':
    main()
