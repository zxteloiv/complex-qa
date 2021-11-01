from typing import Dict
from collections import defaultdict
from functools import partial
import torch.nn
import pickle
import sys
import os
import os.path as osp
import logging
from tqdm import tqdm
from trialbot.utils.move_to_device import move_to_device
from trialbot.training import TrialBot, Registry, Events
from trialbot.data import NSVocabulary
from trialbot.training.updater import Updater
from trialbot.training.hparamset import HyperParamSet
from trialbot.utils.root_finder import find_root

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

from utils.select_optim import select_optim
from trialbot.data.iterators import RandomIterator
import datasets.cfq
import datasets.cfq_translator as cfq_translator
datasets.cfq.install_cfq_to_trialbot()
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_parsed_qa_datasets(Registry._datasets)
import datasets.cg_bundle_translator as sql_translator


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
    p.tgt_ns = datasets.cfq_translator.UNIFIED_TREE_NS

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
    p.tree_encoder_weight_norm = False

    p.bilinear_rank = 1
    p.bilinear_pool = p.hidden_sz // p.num_heads
    p.bilinear_linear = True
    p.bilinear_bias = True

    p.num_re0_layer = 6

    p.use_attn_residual_norm = True

    p.tree_self_attn = 'seq_mha'    # seq_mha, generalized_dot_product

    # ----------- end of tree settings -------------

    p.exact_token_predictor = "quant" # linear, mos, quant
    p.num_exact_token_mixture = 1
    p.exact_token_quant_criterion = "dot_product"
    p.masked_exact_token_training = True
    p.masked_exact_token_testing = True

    p.rule_scorer = "triple_inner_product" # heuristic, mlp, (triple|concat|add_inner_product)
    return p


@Registry.hparamset()
def sql_pda():
    p = HyperParamSet.common_settings(find_root())
    p.TRAINING_LIMIT = 100
    p.OPTIM = "adabelief"
    p.batch_sz = 32
    p.WEIGHT_DECAY = 1e-6
    p.ADAM_LR = 1e-3
    p.ADAM_BETAS = (0.9, 0.999)
    p.optim_kwargs = {"rectify": False}
    p.GRAD_CLIPPING = .2    # grad norm required to be <= 2

    p.src_ns = 'sent'
    p.tgt_ns = datasets.cfq_translator.UNIFIED_TREE_NS

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
    p.tree_encoder_weight_norm = False

    p.bilinear_rank = 1
    p.bilinear_pool = 32
    p.bilinear_linear = True
    p.bilinear_bias = True

    p.num_re0_layer = 6

    p.use_attn_residual_norm = True

    p.tree_self_attn = 'seq_mha'    # seq_mha, generalized_dot_product

    # ----------- end of tree settings -------------

    p.exact_token_predictor = "quant" # linear, mos, quant
    p.num_exact_token_mixture = 1
    p.exact_token_quant_criterion = "dot_product"
    p.masked_exact_token_training = True
    p.masked_exact_token_testing = True

    p.rule_scorer = "triple_inner_product"  # heuristic, mlp, (triple|concat|add_inner_product)
    return p


def get_tutor_from_train_set(p, vocab, train_set, dataset_name: str):
    builder = cfq_translator.CFQTutorBuilder() if 'cfq_' in dataset_name else sql_translator.SQLTutorBuilder()
    builder.index_with_vocab(vocab)
    tutor_repr = builder.batch_tensor([builder.to_tensor(example) for example in tqdm(train_set)])
    return tutor_repr


def build_tutor_objects(tgt_ns, vocab, tutor_repr):
    from models.neural_pda.token_tutor import ExactTokenTutor
    from models.neural_pda.grammar_tutor import GrammarTutor
    from utils.preprocessing import nested_list_numbers_to_tensors
    ns_s, ns_et = tgt_ns
    token_map: defaultdict
    grammar_map: defaultdict
    token_map, grammar_map = map(tutor_repr.get, ('token_map', 'grammar_map'))
    ett = ExactTokenTutor(vocab.get_vocab_size(ns_s), vocab.get_vocab_size(ns_et), token_map, False)

    ordered_lhs, ordered_rhs_options = list(zip(*grammar_map.items()))
    ordered_rhs_options = nested_list_numbers_to_tensors(ordered_rhs_options)
    gt = GrammarTutor(vocab.get_vocab_size(ns_s), ordered_lhs, ordered_rhs_options)

    return ett, gt


def init_tailored_tutor(p, vocab, *, dataset_name: str):
    os.makedirs(os.path.join(p.DATA_PATH, 'npda-tutors-dump'), exist_ok=True)
    tutor_dump_name = os.path.join(p.DATA_PATH, 'npda-tutors-dump', f"tt_{dataset_name}.pkl")

    if os.path.exists(tutor_dump_name):
        logging.getLogger(__name__).info(f"Use the existing tutor dump... {tutor_dump_name}")
        tutor_repr = pickle.load(open(tutor_dump_name, 'rb'))
    else:
        logging.getLogger(__name__).info(f"Build the tutor dump for the first time...")
        tutor_repr = get_tutor_from_train_set(p, vocab, Registry.get_dataset(dataset_name)[0], dataset_name)

        if tutor_dump_name is not None:
            logging.getLogger(__name__).info(f"Dump the tutor information... {tutor_dump_name}")
            pickle.dump(tutor_repr, open(tutor_dump_name, 'wb'))

    return build_tutor_objects(p.tgt_ns, vocab, tutor_repr)


def get_tree_encoder(p, vocab):
    import models.neural_pda.partial_tree_encoder as partial_tree
    from models.modules.decomposed_bilinear import DecomposedBilinear

    # embedding will be transformed into hid size with lhs_symbol_mapper,
    # thus tree encoder input will be hid_sz by default
    if p.tree_encoder == 'lstm':
        tree_encoder = partial_tree.TopDownLSTMEncoder(
            p.emb_sz, p.hidden_sz, p.hidden_sz // p.num_heads,
            dropout=p.dropout,
        )

    elif p.tree_encoder == 'bilinear_tree_lstm':
        tree_encoder = partial_tree.TopDownBilinearLSTMEncoder(
            p.emb_sz, p.hidden_sz, p.bilinear_rank, p.bilinear_pool,
            use_linear=p.bilinear_linear, use_bias=p.bilinear_bias,
            dropout=p.dropout,
        )

    elif p.tree_encoder.startswith('re_zero'):
        assert p.emb_sz == p.hidden_sz, "re0-net requires the embedding and hidden sizes are equal"
        if p.tree_encoder.endswith('bilinear'):
            bilinear_mod = DecomposedBilinear(
                p.emb_sz, p.hidden_sz, p.hidden_sz, p.bilinear_rank, p.bilinear_pool,
                use_linear=p.bilinear_linear, use_bias=p.bilinear_bias,
            )
            if p.tree_encoder_weight_norm:
                # apply weight norm per-pool
                bilinear_mod = torch.nn.utils.weight_norm(bilinear_mod, 'w_o', dim=0)
            layer_encoder = partial_tree.SingleStepBilinear(bilinear_mod)

        elif p.tree_encoder.endswith('dot_prod'):
            layer_encoder = partial_tree.SingleStepDotProd()
        else:
            raise NotImplementedError

        tree_encoder = partial_tree.ReZeroEncoder(num_layers=p.num_re0_layer, layer_encoder=layer_encoder)

    else:
        raise NotImplementedError

    return tree_encoder


def get_rule_scorer(p, vocab):
    from torch import nn
    from models.neural_pda.rule_scorer import MLPScorerWrapper, HeuristicMLPScorerWrapper
    from models.neural_pda.rule_scorer import GeneralizedInnerProductScorer, ConcatInnerProductScorer
    from models.neural_pda.rule_scorer import AddInnerProductScorer
    from models.modules.container import MultiInputsSequential
    from models.modules.variational_dropout import VariationalDropout
    from allennlp.nn.activations import Activation

    if p.rule_scorer == "heuristic":
        rule_scorer = HeuristicMLPScorerWrapper(MultiInputsSequential(
            nn.Linear(p.hidden_sz + p.hidden_sz * 2 + p.hidden_sz * 2, p.hidden_sz),
            VariationalDropout(p.dropout),
            nn.LayerNorm(p.hidden_sz // p.num_heads),
            nn.Tanh(),
            nn.Linear(p.hidden_sz // p.num_heads, p.hidden_sz),
            VariationalDropout(p.dropout),
            nn.LayerNorm(p.hidden_sz),
            Activation.by_name('mish')(),
            nn.Linear(p.hidden_sz, 1)
        ))
    elif p.rule_scorer == "triple_inner_product":
        rule_scorer = GeneralizedInnerProductScorer()
    elif p.rule_scorer == "concat_inner_product":
        rule_scorer = ConcatInnerProductScorer(MultiInputsSequential(
            nn.Linear(p.hidden_sz * 2, p.hidden_sz),
            nn.Tanh(),
        ))
    elif p.rule_scorer == "add_inner_product":
        rule_scorer = AddInnerProductScorer(p.hidden_sz)
    else:
        rule_scorer = MLPScorerWrapper(MultiInputsSequential(
            nn.Linear(p.hidden_sz + p.hidden_sz * 2, p.hidden_sz // p.num_heads),
            VariationalDropout(p.dropout),
            nn.LayerNorm(p.hidden_sz // p.num_heads),
            nn.Tanh(),
            nn.Linear(p.hidden_sz // p.num_heads, p.hidden_sz),
            VariationalDropout(p.dropout),
            nn.LayerNorm(p.hidden_sz),
            Activation.by_name('mish')(),
            nn.Linear(p.hidden_sz, 1)
        ))

    return rule_scorer


def get_model(p, vocab: NSVocabulary, *, dataset_name: str):
    from torch import nn
    from models.neural_pda.seq2pda import Seq2PDA
    from models.neural_pda.npda import NeuralPDA
    from models.base_s2s.stacked_encoder import StackedEncoder
    from models.modules.attention_wrapper import get_wrapped_attention
    from models.modules.quantized_token_predictor import QuantTokenPredictor
    from models.base_s2s.stacked_rnn_cell import StackedRNNCell, StackedLSTMCell
    from models.modules.sym_typed_rnn_cell import SymTypedRNNCell
    from models.modules.container import MultiInputsSequential, UnpackedInputsSequential, SelectArgsById
    from models.modules.mixture_softmax import MoSProjection
    from allennlp.nn.activations import Activation

    from models.transformer.encoder import TransformerEncoder
    from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder

    if p.encoder == "lstm":
        enc_cls = lambda floor: LstmSeq2SeqEncoder(p.enc_sz, p.enc_sz, bidirectional=False)
    elif p.encoder == "transformer":
        enc_cls = lambda floor: TransformerEncoder(
            input_dim=p.enc_sz,
            hidden_dim=p.enc_sz,
            num_layers=1,
            num_heads=p.num_heads,
            feedforward_hidden_dim=p.enc_sz,
            feedforward_dropout=p.dropout,
            residual_dropout=p.dropout,
            attention_dropout=0.,
            use_positional_embedding=(floor == 0),
        )
    elif p.encoder == "bilstm":
        enc_cls = lambda floor: LstmSeq2SeqEncoder(p.enc_sz if floor == 0 else p.enc_sz * 2, # bidirectional
                                                   p.enc_sz, bidirectional=True)
    else:
        raise NotImplementedError

    encoder = StackedEncoder([enc_cls(floor) for floor in range(p.num_enc_layers)], input_dropout=p.dropout)
    emb_src = nn.Embedding(vocab.get_vocab_size(p.src_ns), embedding_dim=p.enc_sz)

    ns_s, ns_et = p.tgt_ns
    emb_s = nn.Embedding(vocab.get_vocab_size(ns_s), p.emb_sz)

    if p.exact_token_predictor == "linear":
        exact_token_predictor = nn.Sequential(
            nn.Linear(p.hidden_sz + p.hidden_sz + p.emb_sz, vocab.get_vocab_size(ns_et)),
        )
    elif p.exact_token_predictor == "quant":
        exact_token_predictor = QuantTokenPredictor(
            vocab.get_vocab_size(ns_et), p.hidden_sz + p.hidden_sz + p.emb_sz,
            quant_criterion=p.exact_token_quant_criterion,
        )
    else:
        exact_token_predictor = MoSProjection(
            p.num_exact_token_mixture, p.hidden_sz + p.hidden_sz + p.emb_sz, vocab.get_vocab_size(ns_et),
        )

    ett, gt = init_tailored_tutor(p, vocab, dataset_name=dataset_name)
    if p.expander_rnn == 'lstm':
        rhs_expander=StackedLSTMCell(p.emb_sz + 1, p.hidden_sz, p.num_expander_layer, dropout=p.dropout)
    else:
        rhs_expander=StackedRNNCell([
            SymTypedRNNCell(input_dim=p.emb_sz + 1 if floor == 0 else p.hidden_sz,
                            output_dim=p.hidden_sz,
                            nonlinearity="tanh")
            for floor in range(p.num_expander_layer)
        ], dropout=p.dropout )

    npda = NeuralPDA(
        symbol_embedding=emb_s,
        lhs_symbol_mapper=MultiInputsSequential(
            nn.Linear(p.emb_sz, p.hidden_sz // p.num_heads),
            nn.LayerNorm(p.hidden_sz // p.num_heads),
            nn.Linear(p.hidden_sz // p.num_heads, p.hidden_sz),
            nn.LayerNorm(p.hidden_sz),  # rule embedding has the hidden_size
            nn.Tanh(),
        ) if p.emb_sz != p.hidden_sz else Activation.by_name('linear')(),   # `linear` returns the identity function
        grammar_tutor=gt,
        rhs_expander=rhs_expander,

        rule_scorer=get_rule_scorer(p, vocab),

        exact_token_predictor=exact_token_predictor,
        token_tutor=ett,

        pre_tree_encoder=get_tree_encoder(p, vocab),
        pre_tree_self_attn=UnpackedInputsSequential(
            get_wrapped_attention(p.tree_self_attn, p.hidden_sz, p.hidden_sz,
                                  num_heads=p.num_heads,
                                  use_linear=p.attn_use_linear,
                                  use_bias=p.attn_use_bias,
                                  use_tanh_activation=p.attn_use_tanh_activation,
                                  ),
            SelectArgsById(0),
        ),
        residual_norm_after_self_attn=nn.LayerNorm(p.hidden_sz) if p.use_attn_residual_norm else None,

        grammar_entry=vocab.get_token_index(p.grammar_entry, ns_s),
        max_derivation_step=p.max_derivation_step,
        masked_exact_token_training=p.masked_exact_token_training,
        masked_exact_token_testing=p.masked_exact_token_testing,
    )

    enc_attn_net = UnpackedInputsSequential(
        get_wrapped_attention(p.enc_attn, p.hidden_sz, encoder.get_output_dim(),
                              num_heads=p.num_heads,
                              use_linear=p.attn_use_linear,
                              use_bias=p.attn_use_bias,
                              use_tanh_activation=p.attn_use_tanh_activation,
                              ),
        SelectArgsById(0),
    )
    enc_attn_mapping = (Activation.by_name('linear')() if p.hidden_sz == encoder.get_output_dim() else
                        nn.Linear(encoder.get_output_dim(), p.hidden_sz))

    model = Seq2PDA(
        encoder=encoder,
        src_embedding=emb_src,
        enc_attn_net=enc_attn_net,
        enc_attn_mapping=enc_attn_mapping,
        npda=npda,
        src_ns=p.src_ns,
        tgt_ns=p.tgt_ns,
        vocab=vocab,
    )

    return model


class PDATrainingUpdater(Updater):
    def __init__(self, bot: TrialBot):
        args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger
        params = model.parameters()
        optim = select_optim(p, params)
        logger.info(f"Using the optimizer {str(optim)}")
        iterator = RandomIterator(len(bot.train_set), p.batch_sz)
        super().__init__(model, iterator, optim, args.device)
        self.clip_grad = p.GRAD_CLIPPING
        self.dataset = bot.train_set
        self.translator = bot.translator

    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_end_of_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()
        indices = next(iterator)
        tensor_list = [self.translator.to_tensor(self.dataset[index]) for index in indices]
        batch = self.translator.batch_tensor(tensor_list)

        if device >= 0:
            batch = move_to_device(batch, device)

        output = model(**batch)
        output['param_update'] = None

        loss = output['loss']
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
        optim.step()
        return output


def main(args=None):
    from utils.trialbot.setup import setup
    if args is None:
        args = setup(seed=2021)

    get_model_fn = partial(get_model, dataset_name=args.dataset)
    bot = TrialBot(trial_name="s2pda_qa", get_model_func=get_model_fn, args=args)

    from utils.trialbot.extensions import collect_garbage, print_hyperparameters, print_models, get_metrics
    bot.add_event_handler(Events.STARTED, print_models, 100)
    bot.add_event_handler(Events.STARTED, print_hyperparameters, 100)
    bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100)
    if not args.test:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trialbot.extensions import end_with_nan_loss
        from utils.trialbot.extensions import evaluation_on_dev_every_epoch
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90, skip_first_epochs=1)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 80)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 80)

        bot.updater = PDATrainingUpdater(bot)

    elif args.debug:
        bot.add_event_handler(Events.ITERATION_COMPLETED, prediction_analysis, 100)

    bot.run()
    return bot


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
