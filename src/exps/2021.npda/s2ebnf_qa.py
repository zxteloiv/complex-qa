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
def cfq_ebnf_qa():
    from trialbot.training.hparamset import HyperParamSet
    from utils.root_finder import find_root
    ROOT = find_root()
    p = HyperParamSet.common_settings(ROOT)
    p.TRAINING_LIMIT = 50
    p.batch_sz = 50
    p.weight_decay = .2

    p.src_ns = 'questionPatternModEntities'
    p.tgt_ns = datasets.cfq_translator.PARSE_TREE_NS
    p.tgt_ns_fi = datasets.cfq_translator.NS_FI

    p.enc_attn = "generalized_bilinear"
    p.dec_hist_attn = "dot_product"
    # transformer requires input embedding equal to hidden size
    p.encoder = "transformer"
    p.emb_sz = 128
    p.hidden_sz = 128
    p.num_heads = 8
    p.attention_dropout = 0.
    p.attn_use_linear = True
    p.attn_use_bias = True
    p.attn_use_tanh_activation = True
    p.num_enc_layers = 2
    p.dropout = .2
    p.decoder = "lstm"
    p.num_expander_layer = 1
    p.default_max_derivation = 100
    p.default_max_expansion = 10
    p.stack_capacity = 100
    p.tied_nonterminal_emb = True
    p.tied_terminal_emb = True
    p.nt_pred_crit = "projection" # projection based or distance based
    p.t_pred_crit = "projection"
    p.grammar_entry = "queryunit"
    return p

def get_model(p, vocab: NSVocabulary):
    from torch import nn
    from models.neural_pda.seq2pda import Seq2PDA
    from models.neural_pda.ebnf_npda import NeuralEBNF
    from models.neural_pda.batched_stack import TensorBatchStack
    from models.modules.stacked_encoder import StackedEncoder
    from models.modules.attention_wrapper import get_wrapped_attention
    from models.modules.quantized_token_predictor import QuantTokenPredictor
    from models.modules.stacked_rnn_cell import StackedLSTMCell
    from models.modules.container import MultiInputsSequential
    from trialbot.data import START_SYMBOL, PADDING_TOKEN, END_SYMBOL

    encoder = StackedEncoder.get_encoder(p)
    emb_src = nn.Embedding(vocab.get_vocab_size(p.src_ns), embedding_dim=p.emb_sz)

    ns_nt, ns_t, ns_et = p.tgt_ns
    emb_nt = nn.Embedding(vocab.get_vocab_size(ns_nt), p.emb_sz)
    emb_t = nn.Embedding(vocab.get_vocab_size(ns_t), p.emb_sz)

    nt_pred = QuantTokenPredictor(
        num_toks=vocab.get_vocab_size(ns_nt),
        tok_dim=p.emb_sz,
        shared_embedding=emb_nt.weight if p.tied_nonterminal_emb else None,
        quant_criterion=p.nt_pred_crit,
    )
    t_pred = QuantTokenPredictor(
        num_toks=vocab.get_vocab_size(ns_t),
        tok_dim=p.emb_sz,
        shared_embedding=emb_t.weight if p.tied_terminal_emb else None,
        quant_criterion=p.t_pred_crit,
    )

    ebnf_pda = NeuralEBNF(
        emb_nonterminals=emb_nt,
        emb_terminals=emb_t,
        num_nonterminals=vocab.get_token_index(ns_nt),
        ebnf_expander=StackedLSTMCell(
            input_dim=p.emb_sz * 2 + 1,
            hidden_dim=p.hidden_sz + 1,  # quant predictor requires input hidden == embedding size
            n_layers=p.num_expander_layer,
            intermediate_dropout=p.dropout,
        ),
        state_transition=None,
        batch_stack=TensorBatchStack(p.batch_sz, p.stack_capacity, 1 + 1),
        predictor_nonterminals=nt_pred,
        predictor_terminals=t_pred,
        start_token_id=vocab.get_token_index(START_SYMBOL, ns_nt),
        ebnf_entrypoint=vocab.get_token_index(p.grammar_entry, ns_nt),
        dropout=p.dropout,
        default_max_derivation=p.default_max_derivation,
        default_max_expansion=p.default_max_expansion,
    )
    enc_attn_net = MultiInputsSequential(
        get_wrapped_attention(p.enc_attn, p.hidden_sz + 1, encoder.get_output_dim(),
                              num_heads=p.num_heads, use_linear=p.attn_use_linear, use_bias=p.attn_use_bias,
                              use_tanh_activation=p.attn_use_tanh_activation),
        nn.Linear(encoder.get_output_dim(), p.hidden_sz + 1),
    )

    model = Seq2PDA(
        encoder=encoder,
        src_embedding=emb_src,
        enc_attn_net=enc_attn_net,
        ebnf_npda=ebnf_pda,
        tok_pad_id=0,
        nt_fi=p.tgt_ns_fi[0],
    )

    return model

def main():
    from utils.trialbot_setup import setup
    args = setup(seed=2021)

    bot = TrialBot(trial_name="s2pda_qa", get_model_func=get_model, args=args)

    from trialbot.training import Events
    @bot.attach_extension(Events.EPOCH_COMPLETED)
    def get_metrics(bot: TrialBot):
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
        bot.add_event_handler(Events.EPOCH_COMPLETED, evaluation_on_dev_every_epoch, 90)

        bot.add_event_handler(Events.ITERATION_COMPLETED, end_with_nan_loss, 100)
        bot.add_event_handler(Events.EPOCH_COMPLETED, every_epoch_model_saver, 100)

        # debug strange errors by inspecting running time, data size, etc.
        if args.debug:
            from utils.trial_bot_extensions import track_pytorch_module_forward_time
            bot.add_event_handler(Events.STARTED, track_pytorch_module_forward_time, 100)

        @bot.attach_extension(Events.ITERATION_COMPLETED)
        def batch_data_size(bot: TrialBot):
            output = bot.state.output
            if output is None:
                return

            print("source_size:", output['source_tokens'].size(),
                  "tree_size:", output['derivation_tree'].size(),
                  "tofi_size:", output['token_fidelity'].size())
    bot.run()

if __name__ == '__main__':
    main()
