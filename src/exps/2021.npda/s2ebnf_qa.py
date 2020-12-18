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
    p.TRAINING_LIMIT = 5
    p.OPTIM = "RAdam"
    p.batch_sz = 32
    p.weight_decay = .2

    p.src_ns = 'questionPatternModEntities'
    p.tgt_ns = datasets.cfq_translator.PARSE_TREE_NS
    p.tgt_ns_fi = datasets.cfq_translator.NS_FI
    p.NS_VOCAB_KWARGS = {"non_padded_namespaces": p.tgt_ns[1:]}

    p.enc_attn = "generalized_dot_product"
    # transformer requires input embedding equal to hidden size
    p.encoder = "transformer"
    p.emb_sz = 128
    p.hidden_sz = 128
    p.num_heads = 8
    p.attn_use_linear = False
    p.attn_use_bias = False
    p.attn_use_tanh_activation = False
    p.num_enc_layers = 2
    p.dropout = .2
    p.decoder = "lstm"
    p.num_expander_layer = 1
    p.default_max_derivation = 100
    p.default_max_expansion = 10
    p.stack_capacity = 100
    p.tied_nonterminal_emb = True
    p.tied_terminal_emb = True
    p.nt_pred_crit = "projection" # distance, projection, dot_product
    p.t_pred_crit = "projection"
    p.grammar_entry = "queryunit"

    p.joint_topology_control = False
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

    topo_pred = None
    dec_out_dim: int = p.hidden_sz + 1
    if not p.joint_topology_control:
        topo_pred = nn.Sequential(
            nn.Linear(p.hidden_sz, 1),
            nn.Sigmoid(),
        )
        dec_out_dim = p.hidden_sz

    ebnf_pda = NeuralEBNF(
        emb_nonterminals=emb_nt,
        emb_terminals=emb_t,
        num_nonterminals=vocab.get_token_index(ns_nt),
        ebnf_expander=StackedLSTMCell(
            input_dim=p.emb_sz * 2 + 1,
            hidden_dim=dec_out_dim,  # quant predictor requires input hidden == embedding size
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
        topology_predictor=topo_pred,
    )
    enc_attn_net = MultiInputsSequential(
        get_wrapped_attention(p.enc_attn, dec_out_dim, encoder.get_output_dim(),
                              num_heads=p.num_heads, use_linear=p.attn_use_linear, use_bias=p.attn_use_bias,
                              use_tanh_activation=p.attn_use_tanh_activation),
        nn.Linear(encoder.get_output_dim(), dec_out_dim),
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

        from utils.trial_bot_extensions import init_tensorboard_writer
        from utils.trial_bot_extensions import write_batch_info_to_tensorboard
        from utils.trial_bot_extensions import close_tensorboard
        # bot.add_event_handler(Events.STARTED, init_tensorboard_writer, 100)
        # bot.add_event_handler(Events.ITERATION_COMPLETED, write_batch_info_to_tensorboard, 100)
        # bot.add_event_handler(Events.COMPLETED, close_tensorboard, 100)

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
    else:
        if args.debug:
            bot.add_event_handler(Events.ITERATION_COMPLETED, prediction_analysis, 100)
    bot.run()

def prediction_analysis(bot: TrialBot):
    # everything is (batch, derivation, rhs_seq - 1)
    # except for lhs is (batch, derivation)
    #
    # { "error": [is_nt_err, nt_err, t_err],
    #   "gold": [mask, out_is_nt, safe_nt_out, safe_t_out],
    #   "all_lhs": derivation_tree[:, :, 0], }
    #
    # preds = (is_nt_prob > 0.5, nt_logits.argmax(dim=-1), t_logits.argmax(dim=-1))
    #
    output = bot.state.output['deliberate_analysis']
    source_tokens = output['source_tokens']
    is_nt_err, nt_err, t_err = output['error']
    mask, out_is_nt, safe_nt_out, safe_t_out = output['gold']
    preds = bot.state.output['preds']
    all_lhs = output['all_lhs']
    print('===' * 30)
    batch_sz, drv_num = all_lhs.size()

    nt_tok = lambda k: bot.vocab.get_token_from_index(k, bot.hparams.tgt_ns[0])
    t_tok = lambda k: bot.vocab.get_token_from_index(k, bot.hparams.tgt_ns[1])

    def interweave_rule(is_nt, nt_toks, t_toks, padding_start):
        # all: (rhs_seq - 1,)
        expansion_id = [str(nt.item()) if cond > 0 else str(t.item())
                        for cond, nt, t in zip(is_nt, nt_toks, t_toks)]
        expansion_tok = [nt_tok(nt.item()) if cond > 0 else t_tok(t.item())
                         for cond, nt, t in zip(is_nt, nt_toks, t_toks)]
        expansion_id.insert(padding_start, '||')
        expansion_tok.insert(padding_start, '||')
        return expansion_id, expansion_tok

    for i in range(batch_sz):
        if mask[i][0][0].item() == 0:
            continue

        if i > 0:
            print('===' * 30)

        print("SRC_TOKS:" + " ".join(bot.vocab.get_token_from_index(tok, bot.hparams.src_ns)
                                     for tok in source_tokens[i].tolist()))

        for j in range(drv_num):
            if mask[i][j][0].item() == 0:
                continue

            if j > 0:
                print('---' * 20)

            padding_start = mask[i][j].sum().item()

            gold_id, gold_tok = interweave_rule(out_is_nt[i][j], safe_nt_out[i][j], safe_t_out[i][j], padding_start)
            print(f'GOLD_ID:  {all_lhs[i][j].item()}   --> {" ".join(gold_id)}')
            print(f'GOLD_TOK: {nt_tok(all_lhs[i][j].item())} --> {" ".join(gold_tok)}')

            pred_medal = []
            for gz, z, nt, t in zip(out_is_nt[i][j], is_nt_err[i][j], nt_err[i][j], t_err[i][j]):
                z, nt, t = list(map(lambda n: 'x' if n > 0 else 'o', (z, nt, t)))
                medal = (z + nt) if gz > 0 else (z + t)
                pred_medal.append(medal)
            pred_medal.insert(padding_start, '|')

            pred_id, pred_tok = interweave_rule(preds[0][i][j], preds[1][i][j], preds[2][i][j], padding_start)
            print(f'PRED_ID:  {all_lhs[i][j].item()}   --> {" ".join(pred_id)}')
            print(f'PRED_TOK: {nt_tok(all_lhs[i][j].item())} --> {" ".join("%s(%s)" % t for t in zip(pred_tok, pred_medal))}')

if __name__ == '__main__':
    main()
