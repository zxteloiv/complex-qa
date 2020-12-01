from trialbot.data import NSVocabulary, PADDING_TOKEN, START_SYMBOL, END_SYMBOL
from torch import nn

def lm_ebnf(p, vocab: NSVocabulary):
    from models.neural_pda.ebnf_npda import NeuralEBNF
    from models.neural_pda.batched_stack import TensorBatchStack
    from models.neural_pda.formal_language_model import EBNFTreeLM
    from models.modules.stacked_rnn_cell import StackedLSTMCell
    from models.modules.quantized_token_predictor import QuantTokenPredictor

    ns_nt, ns_t, ns_et = p.ns
    emb_nt = nn.Embedding(vocab.get_vocab_size(ns_nt), p.emb_sz, max_norm=p.nt_emb_max_norm)
    emb_t = nn.Embedding(vocab.get_vocab_size(ns_t), p.emb_sz, max_norm=p.t_emb_max_norm)

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

    pda = NeuralEBNF(
        emb_nonterminals=emb_nt,
        emb_terminals=emb_t,
        num_nonterminals=vocab.get_token_index(ns_nt),
        ebnf_expander=StackedLSTMCell(
            input_dim=p.emb_sz * 2 + 1,
            hidden_dim=p.hidden_dim + 1,    # quant predictor requires input hidden == embedding size
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
    )

    model = EBNFTreeLM(pda, tok_pad_id=vocab.get_token_index(PADDING_TOKEN, ns_nt), nt_fi=p.ns_fi[0])
    return model
