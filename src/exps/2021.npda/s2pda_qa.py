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

if __name__ == '__main__':
    sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

from utils.trialbot.setup_cli import setup
from utils.select_optim import select_optim
from trialbot.data.iterators import RandomIterator
import shujuji.cfq
import shujuji.cfq_translator as cfq_translator
shujuji.cfq.install_cfq_to_trialbot()
import shujuji.cg_bundle as cg_bundle
cg_bundle.install_parsed_qa_datasets(Registry._datasets)
import shujuji.cg_bundle_translator as sql_translator

import s2pda_hparams


def get_tutor_from_train_set(vocab, train_set, dataset_name: str):
    builder = cfq_translator.CFQTutorBuilder() if 'cfq_' in dataset_name else sql_translator.SQLTutorBuilder()
    builder.index_with_vocab(vocab)
    tutor_repr = builder.batch_tensor([builder.to_tensor(example) for example in tqdm(train_set)])
    return tutor_repr


def build_tutor_objects(tgt_ns, vocab, tutor_repr):
    from models.neural_pda.grammar_tutor import GrammarTutor
    from utils.preprocessing import nested_list_numbers_to_tensors
    grammar_map: defaultdict = tutor_repr.get('grammar_map')
    ordered_lhs, ordered_rhs_options = list(zip(*grammar_map.items()))
    ordered_rhs_options = nested_list_numbers_to_tensors(ordered_rhs_options)
    gt = GrammarTutor(vocab.get_vocab_size(tgt_ns), ordered_lhs, ordered_rhs_options)
    return gt


def get_tree_encoder(p, vocab):
    import models.neural_pda.partial_tree_encoder as partial_tree
    from models.modules.decomposed_bilinear import DecomposedBilinear
    from allennlp.nn.activations import Activation

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
            layer_encoder = partial_tree.SingleStepBilinear(bilinear_mod)
        elif p.tree_encoder.endswith('dot_prod'):
            layer_encoder = partial_tree.SingleStepDotProd()
        else:
            raise NotImplementedError

        tree_encoder = partial_tree.ReZeroEncoder(
            num_layers=p.num_re0_layer,
            layer_encoder=layer_encoder,
            activation=Activation.by_name(getattr(p, 're0_activation', 'linear'))(),
        )

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
            Activation.by_name('tanh')(),
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
    elif p.rule_scorer == "gated_add_inner_product":
        rule_scorer = AddInnerProductScorer(p.hidden_sz, use_gated_add=True)
    else:
        rule_scorer = MLPScorerWrapper(MultiInputsSequential(
            nn.Linear(p.hidden_sz + p.hidden_sz * 2, p.hidden_sz // p.num_heads),
            VariationalDropout(p.dropout),
            nn.LayerNorm(p.hidden_sz // p.num_heads),
            nn.Tanh(),
            nn.Linear(p.hidden_sz // p.num_heads, p.hidden_sz),
            VariationalDropout(p.dropout),
            nn.LayerNorm(p.hidden_sz),
            Activation.by_name('tanh')(),
            nn.Linear(p.hidden_sz, 1),
        ))

    return rule_scorer


def get_model(p, vocab: NSVocabulary):
    from torch import nn
    from models.neural_pda.seq2pda import Seq2PDA
    from models.neural_pda.npda import NeuralPDA
    from models.base_s2s.encoder_stacker import EncoderStacker
    from models.modules.attention_wrapper import get_wrapped_attention
    from models.base_s2s.stacked_rnn_cell import StackedRNNCell, StackedLSTMCell
    from models.modules.container import MultiInputsSequential, UnpackedInputsSequential, SelectArgsById
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
            feedforward_dropout=0.,
            residual_dropout=0.,
            attention_dropout=0.,
            use_positional_embedding=(floor == 0),
        )
    elif p.encoder == "bilstm":
        enc_cls = lambda floor: LstmSeq2SeqEncoder(p.enc_sz if floor == 0 else p.enc_sz * 2, # bidirectional
                                                   p.enc_sz, bidirectional=True)
    else:
        raise NotImplementedError

    encoder = EncoderStacker([enc_cls(floor) for floor in range(p.num_enc_layers)], input_dropout=0.)
    emb_src = nn.Embedding(vocab.get_vocab_size(p.src_ns), embedding_dim=p.enc_sz)
    emb_s = nn.Embedding(vocab.get_vocab_size(p.tgt_ns), p.emb_sz)
    if getattr(p, 'init_embedding_by_kaiming_normal', False):
        nn.init.kaiming_normal_(emb_src.weight)
        nn.init.kaiming_normal_(emb_s.weight)

    npda = NeuralPDA(
        symbol_embedding=emb_s,
        lhs_symbol_mapper=MultiInputsSequential(
            nn.Linear(p.emb_sz, p.hidden_sz // p.num_heads),
            nn.LayerNorm(p.hidden_sz // p.num_heads),
            nn.Linear(p.hidden_sz // p.num_heads, p.hidden_sz),
            nn.LayerNorm(p.hidden_sz),  # rule embedding has the hidden_size
            nn.Tanh(),
        ) if p.emb_sz != p.hidden_sz else Activation.by_name('linear')(),   # `linear` returns the identity function
        grammar_tutor=None,
        rhs_expander=StackedLSTMCell(p.emb_sz + 1, p.hidden_sz, p.num_expander_layer, dropout=p.dropout),
        rule_scorer=get_rule_scorer(p, vocab),
        tree_encoder=get_tree_encoder(p, vocab),
        tree_self_attn=UnpackedInputsSequential(
            get_wrapped_attention(p.tree_self_attn, p.hidden_sz, p.hidden_sz,
                                  num_heads=p.num_heads,
                                  use_linear=p.attn_use_linear,
                                  use_bias=p.attn_use_bias,
                                  use_tanh_activation=p.attn_use_tanh_activation,
                                  ),
            SelectArgsById(0),
        ),
        residual_norm_after_self_attn=nn.LayerNorm(p.hidden_sz) if p.use_attn_residual_norm else None,

        grammar_entry=vocab.get_token_index(p.grammar_entry, p.tgt_ns),
        max_derivation_step=p.max_derivation_step,
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
        repr_loss_lambda=getattr(p, 'repr_loss_lambda', 0.)
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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_grad)
        optim.step()
        return output


def make_trialbot(args=None, print_details: bool = False, epoch_eval: bool = False,
                  train_metric_pref: str = 'Training Metrics: ',
                  ):
    if args is None:
        args = setup(seed=2021)

    bot = TrialBot(trial_name="s2pda_qa", get_model_func=get_model, args=args)

    from utils.trialbot.extensions import collect_garbage, print_hyperparameters, print_models, get_metrics
    if print_details:
        bot.add_event_handler(Events.STARTED, print_models, 100)
        bot.add_event_handler(Events.STARTED, print_hyperparameters, 100)

    bot.add_event_handler(Events.STARTED, update_grammar_tutor, 100)
    bot.add_event_handler(Events.EPOCH_COMPLETED, get_metrics, 100, prefix=train_metric_pref)
    if not args.test:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trialbot.extensions import end_with_nan_loss
        if epoch_eval:
            from utils.trialbot.extensions import evaluation_on_dev_every_epoch
            bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90, skip_first_epochs=0)
            bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90, skip_first_epochs=0, on_test_data=True)

        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, collect_garbage, 80)
        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, collect_garbage, 80)

        bot.updater = PDATrainingUpdater(bot)

    return bot


def update_grammar_tutor(bot: TrialBot):
    args, p, vocab, logger = bot.args, bot.hparams, bot.vocab, bot.logger
    dataset_name = args.dataset
    from models.neural_pda.seq2pda import Seq2PDA
    s2pda: Seq2PDA = bot.model
    logger.info(f"Update the tutor ...")
    tutor_repr = get_tutor_from_train_set(vocab, bot.train_set, dataset_name)
    gt = build_tutor_objects(p.tgt_ns, vocab, tutor_repr)
    if args.device >= 0:
        gt = gt.cuda(args.device)
    s2pda.npda._gt = gt


if __name__ == '__main__':
    make_trialbot(print_details=True, epoch_eval=True).run()
