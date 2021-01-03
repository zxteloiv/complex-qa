from typing import Dict
import sys, os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import logging

from trialbot.training import TrialBot, Registry
from trialbot.data import NSVocabulary
from trialbot.training.updater import TrainingUpdater, TestingUpdater
from trialbot.utils.move_to_device import move_to_device

import datasets.cfq
import datasets.cfq_translator

@Registry.hparamset()
def cfq_pda():
    from trialbot.training.hparamset import HyperParamSet
    from utils.root_finder import find_root
    ROOT = find_root()
    p = HyperParamSet.common_settings(ROOT)
    p.TRAINING_LIMIT = 5
    p.OPTIM = "RAdam"
    p.batch_sz = 64
    p.weight_decay = .2

    p.src_ns = 'questionPatternModEntities'
    p.tgt_ns = datasets.cfq_translator.UNIFIED_TREE_NS

    p.enc_attn = "bilinear"
    # transformer requires input embedding equal to hidden size
    p.encoder = "bilstm"
    p.emb_sz = 128
    p.hidden_sz = 128
    p.num_heads = 8
    p.attn_use_linear = False
    p.attn_use_bias = False
    p.attn_use_tanh_activation = False
    p.num_enc_layers = 2
    p.dropout = .2
    p.num_expander_layer = 2
    p.max_derivation_step = 500
    p.max_expansion_len = 11
    p.tied_symbol_emb = True
    p.symbol_quant_criterion = "projection"
    p.grammar_entry = "queryunit"

    p.num_exact_token_mixture = 1

    return p

def get_grammar_tutor(vocab, ns_symbol):
    from models.neural_pda.grammar_tutor import GrammarTutorForGeneration
    grammar_dataset, _, _ = datasets.cfq.sparql_pattern_grammar()
    grammar_translator = datasets.cfq_translator.UnifiedLarkTranslator()
    grammar_translator.index_with_vocab(vocab)
    rule_list = [grammar_translator.to_tensor(r) for r in grammar_dataset]
    rule_repr = grammar_translator.batch_tensor(rule_list)
    gt = GrammarTutorForGeneration(vocab.get_vocab_size(ns_symbol), {int(k): v for k, v in rule_repr.items()})
    return gt

def get_model(p, vocab: NSVocabulary):
    import torch
    from torch import nn
    from models.neural_pda.seq2pda import Seq2PDA
    from models.neural_pda.npda import NeuralPDA
    from models.neural_pda.batched_stack import TensorBatchStack
    from models.neural_pda.grammar_tutor import GrammarTutorForGeneration
    from models.modules.stacked_encoder import StackedEncoder
    from models.modules.attention_wrapper import get_wrapped_attention
    from models.modules.quantized_token_predictor import QuantTokenPredictor
    from models.modules.stacked_rnn_cell import StackedRNNCell
    from models.modules.sym_typed_rnn_cell import SymTypedRNNCell
    from models.modules.container import MultiInputsSequential
    from models.modules.mixture_softmax import MoSProjection
    from models.modules.attention_composer import ClassicMLPComposer, CatComposer, AddComposer
    from trialbot.data import START_SYMBOL, PADDING_TOKEN, END_SYMBOL
    from allennlp.modules.matrix_attention import DotProductMatrixAttention
    from allennlp.nn.activations import Activation

    encoder = StackedEncoder.get_encoder(p)
    emb_src = nn.Embedding(vocab.get_vocab_size(p.src_ns), embedding_dim=p.emb_sz)

    ns_s, ns_et = p.tgt_ns
    emb_s = nn.Embedding(vocab.get_vocab_size(ns_s), p.emb_sz)
    symbol_predictor = QuantTokenPredictor(
        num_toks=vocab.get_vocab_size(ns_s),
        tok_dim=p.emb_sz,
        shared_embedding=emb_s.weight if p.tied_symbol_emb else None,
        quant_criterion=p.symbol_quant_criterion,
    )

    npda = NeuralPDA(
        symbol_embedding=emb_s,
        grammar_tutor=get_grammar_tutor(vocab, ns_s),
        rhs_expander=StackedRNNCell(
            [
                SymTypedRNNCell(input_dim=p.emb_sz if floor == 0 else p.hidden_sz, output_dim=p.hidden_sz, nonlinearity="mish")
                for floor in range(p.num_expander_layer)
            ],
            p.emb_sz, p.hidden_sz, p.num_expander_layer, intermediate_dropout=p.dropout
        ),
        stack=TensorBatchStack(p.batch_sz, p.max_derivation_step, item_size=1, dtype=torch.long),
        symbol_predictor=symbol_predictor,
        exact_form_predictor=MoSProjection(
            p.num_exact_token_mixture, p.hidden_sz + p.emb_sz, vocab.get_vocab_size(ns_et), output_semantics="probs"
        ),
        query_attention_composer=ClassicMLPComposer(encoder.get_output_dim(), p.hidden_sz, p.hidden_sz),
        grammar_entry=vocab.get_token_index(p.grammar_entry, ns_s),
        max_derivation_step=p.max_derivation_step,
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
        max_expansion_len=p.max_expansion_len,
        src_ns=p.src_ns,
        tgt_ns=p.tgt_ns,
        vocab=vocab,
    )

    return model

class Seq2PDATrainingUpdater(TrainingUpdater):
    def update_epoch(self):
        model, optim, iterator = self._models[0], self._optims[0], self._iterators[0]

        if iterator.is_new_epoch:
            self.stop_epoch()

        device = self._device
        model.train()
        optim.zero_grad()
        batch  = next(iterator)

        if device >= 0:
            batch = move_to_device(batch, device)

        output = model(**batch)
        optim.step()

        return output

def main():
    from utils.trialbot_setup import setup
    args = setup(seed=2021)

    bot = TrialBot(trial_name="s2pda_qa", get_model_func=get_model, args=args)

    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def get_metric(bot: TrialBot):
        import json
        print(json.dumps(bot.model.get_metric(reset=True)))

    @bot.attach_extension(Events.STARTED)
    def print_models(bot: TrialBot):
        print(str(bot.models))

    from trialbot.training import Events
    if not args.test:
        # --------------------- Training -------------------------------
        from trialbot.training.extensions import every_epoch_model_saver
        from utils.trial_bot_extensions import end_with_nan_loss
        from utils.trial_bot_extensions import evaluation_on_dev_every_epoch
        from utils.trial_bot_extensions import save_model_every_num_iters
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)
        bot.add_event_handler(Events.ITERATION_COMPLETED, save_model_every_num_iters, 100, interval=100)

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

        bot.updater = Seq2PDATrainingUpdater.from_bot(bot)

    else:
        bot.add_event_handler(Events.ITERATION_COMPLETED, prediction_analysis, 100)
    bot.run()

def prediction_analysis(bot: TrialBot):
    output = bot.state.output
    if output is None:
        return

    output = bot.model.make_human_readable_output(output)
    batch_src = output['source_tokens']
    batch_pred = output['predicted_tokens']
    batch_gold = output['target_tokens']

    for src, pred, gold in zip(batch_src, batch_pred, batch_gold):
        print('---' * 30)
        print("SRC:  " + " ".join(src))
        print("PRED: " + " ".join(pred))
        print("GOLD: " + " ".join(gold))

if __name__ == '__main__':
    main()
