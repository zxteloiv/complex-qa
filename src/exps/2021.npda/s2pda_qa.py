from typing import Dict
import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import logging
from collections import defaultdict

from trialbot.utils.move_to_device import move_to_device
import torch.nn
from trialbot.training import TrialBot, Registry
from trialbot.data import NSVocabulary
from trialbot.training.updater import Updater

import datasets.cfq
import datasets.cfq_translator as cfq_translator
datasets.cfq.install_cfq_to_trialbot()
import datasets.comp_gen_bundle as cg_bundle
cg_bundle.install_sql_qa_datasets(Registry._datasets)
cg_bundle.install_qa_datasets(Registry._datasets)
import datasets.cg_bundle_translator as sql_translator

@Registry.hparamset()
def cfq_pda():
    from trialbot.training.hparamset import HyperParamSet
    from utils.root_finder import find_root
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

    p.return_attn_weights = False

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
def cfq_simp():
    p = cfq_pda()
    p.enc_sz = 128
    p.num_enc_layers = 2

    p.return_attn_weights = False

    p.emb_sz = 128
    p.hidden_sz = 128
    p.num_re0_layer = 6
    p.num_expander_layer = 1
    # p.expander_rnn = 'lstm'   # not implemented yet for inputs requiring broadcasting
    p.GRAD_CLIPPING = .2    # grad norm required to be <= 2
    p.ADAM_BETAS = (0.9, 0.999)
    p.WEIGHT_DECAY = .1
    p.bilinear_rank = 1
    p.bilinear_pool = 32
    p.dropout = 1e-4
    p.rule_scorer = "triple_inner_product"

    p.masked_exact_token_training = True
    p.masked_exact_token_testing = True
    return p

@Registry.hparamset()
def sql_pda():
    p = cfq_pda()
    # p.OPTIM = 'adabelief'
    # p.optim_kwargs = {}
    p.return_attn_weights = False
    p.batch_sz = 16
    p.src_ns = 'sent'
    p.tgt_ns = datasets.cfq_translator.UNIFIED_TREE_NS
    p.TRAINING_LIMIT = 100
    p.encoder = "transformer"
    p.enc_attn = "seq_mha"
    p.enc_sz = 128
    p.num_enc_layers = 2
    p.emb_sz = 128
    p.hidden_sz = 128
    p.num_re0_layer = 6
    p.num_expander_layer = 1
    # p.expander_rnn = 'lstm'   # not implemented yet for inputs requiring broadcasting
    p.GRAD_CLIPPING = .2    # grad norm
    p.ADAM_BETAS = (0.9, 0.999)
    p.WEIGHT_DECAY = 1e-3
    p.bilinear_rank = 1
    p.bilinear_pool = 8
    p.num_heads = 4
    p.dropout = .2
    p.ADAM_LR = 1e-3
    p.rule_scorer = "add_inner_product"
    p.masked_exact_token_training = True
    p.masked_exact_token_testing = True
    p.attn_use_tanh_activation = True
    return p

def get_tailored_tutor(p, vocab, *, dataset_name: str):
    from tqdm import tqdm
    import pickle
    from models.neural_pda.token_tutor import ExactTokenTutor
    from models.neural_pda.grammar_tutor import GrammarTutor
    from utils.preprocessing import nested_list_numbers_to_tensors

    os.makedirs(os.path.join(p.DATA_PATH, 'npda-tutors-dump'), exist_ok=True)
    tutor_dump_name = os.path.join(p.DATA_PATH, 'npda-tutors-dump', f"tt_{dataset_name}.pkl")

    train_set = Registry.get_dataset(dataset_name)[0]
    if os.path.exists(tutor_dump_name):
        logging.getLogger(__name__).info(f"Use the existing tutor dump... {tutor_dump_name}")
        tutor_repr = pickle.load(open(tutor_dump_name, 'rb'))
    else:
        logging.getLogger(__name__).info(f"Build the tutor dump for the first time...")

        builder = cfq_translator.CFQTutorBuilder() if 'cfq_' in dataset_name else sql_translator.SQLTutorBuilder()
        builder.index_with_vocab(vocab)
        tutor_repr = builder.batch_tensor([builder.to_tensor(example) for example in tqdm(train_set)])

        if tutor_dump_name is not None:
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

def get_model(p, vocab: NSVocabulary, *, dataset_name: str):
    from torch import nn
    from models.neural_pda.seq2pda import Seq2PDA
    from models.neural_pda.npda import NeuralPDA
    from models.neural_pda.rule_scorer import MLPScorerWrapper, HeuristicMLPScorerWrapper
    from models.neural_pda.rule_scorer import GeneralizedInnerProductScorer, ConcatInnerProductScorer
    from models.neural_pda.rule_scorer import AddInnerProductScorer
    from models.modules.stacked_encoder import StackedEncoder
    from models.modules.attention_wrapper import get_wrapped_attention
    from models.modules.quantized_token_predictor import QuantTokenPredictor
    from models.modules.stacked_rnn_cell import StackedRNNCell, StackedLSTMCell
    from models.modules.sym_typed_rnn_cell import SymTypedRNNCell
    from models.modules.container import MultiInputsSequential, UnpackedInputsSequential, SelectArgsById
    from models.modules.mixture_softmax import MoSProjection
    from models.modules.variational_dropout import VariationalDropout
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

    ett, gt = get_tailored_tutor(p, vocab, dataset_name=dataset_name)

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

    if p.expander_rnn == 'lstm':
        rhs_expander=StackedLSTMCell(p.emb_sz + 3, p.hidden_sz, p.num_expander_layer, intermediate_dropout=p.dropout)
    else:
        rhs_expander=StackedRNNCell(
            [
                SymTypedRNNCell(input_dim=p.emb_sz + 3 if floor == 0 else p.hidden_sz,
                                output_dim=p.hidden_sz,
                                nonlinearity="tanh")
                for floor in range(p.num_expander_layer)
            ],
            p.emb_sz + 3, p.hidden_sz, p.num_expander_layer, intermediate_dropout=p.dropout
        )

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
        rule_scorer=rule_scorer,
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
        return_attn_weights=p.return_attn_weights,
    )

    return model

class PDATrainingUpdater(Updater):
    def __init__(self, models, iterators, optims, device=-1, clip_grad: float = 0., param_update=False):
        super().__init__(models, iterators, optims, device)
        self.clip_grad = clip_grad
        self.param_update = param_update

    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]
        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()
        batch: Dict[str, torch.Tensor] = next(iterator)

        if device >= 0:
            batch = move_to_device(batch, device)

        output = model(**batch)
        output['param_update'] = None
        if self.param_update:
            output['param_update'] =  {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters()
            }

        loss = output['loss']
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
        optim.step()

        if self.param_update:
            for name, param in model.named_parameters():
                output['param_update'][name].sub_(param.detach().cpu())

        return output

    @classmethod
    def from_bot(cls, bot: TrialBot) -> 'PDATrainingUpdater':
        from utils.maybe_random_iterator import MaybeRandomIterator
        from models.neural_pda.seq2pda import Seq2PDA
        model: Seq2PDA
        args, p, model, logger = bot.args, bot.hparams, bot.model, bot.logger

        params = model.parameters()
        from utils.select_optim import select_optim
        optim = select_optim(p, params)
        logger.info(f"Using the optimizer {str(optim)}")

        repeat_iter = shuffle_iter = not args.debug
        iterator = MaybeRandomIterator(bot.train_set, p.batch_sz, bot.translator, shuffle=shuffle_iter, repeat=repeat_iter)
        if args.debug and args.skip:
            iterator.reset(args.skip)

        updater = cls(model, iterator, optim, args.device, p.GRAD_CLIPPING, param_update=True)
        return updater

def main():
    from utils.trialbot_setup import setup
    args = setup(seed=2021)

    from functools import partial
    get_model_fn = partial(get_model, dataset_name=args.dataset)

    bot = TrialBot(trial_name="s2pda_qa", get_model_func=get_model_fn, args=args)

    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def get_metric(bot: TrialBot):
        import json
        print(json.dumps(bot.model.get_metric(reset=True, diagnosis=True), sort_keys=True))

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
        # bot.add_event_handler(Events.STARTED, init_tensorboard_writer, 100, interval=4, histogram_interval=100)
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
