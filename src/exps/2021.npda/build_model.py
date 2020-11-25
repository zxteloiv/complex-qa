from trialbot.data import NSVocabulary, PADDING_TOKEN, START_SYMBOL, END_SYMBOL
from torch import nn

def lm_npda(p, vocab: NSVocabulary):
    from models.neural_pda.formal_language_model import NPDAFLM
    from models.neural_pda.npda import NeuralPDA, NTDecoder
    from models.neural_pda.batched_stack import BatchStack, TensorBatchStack, ListedBatchStack
    from models.neural_pda.npda_cell import TRNNPDACell, LSTMPDACell
    from models.modules.stacked_rnn_cell import StackedRNNCell, RNNType
    from models.modules.mixture_softmax import MoSProjection
    from models.neural_pda.codebook import VanillaCodebook, SplitCodebook
    tgt_ns = getattr(p, 'target_namespace', 'sparqlPattern')

    pda_type = getattr(p, 'pda_type', 'trnn')
    if pda_type == 'lstm':
        pda = LSTMPDACell(p.token_dim, p.stack_dim, p.hidden_dim)
    else:
        pda = TRNNPDACell(p.token_dim, p.stack_dim, p.hidden_dim)

    codebook_type = getattr(p, 'codebook', 'vanilla')
    if codebook_type == "vanilla":
        codebook = VanillaCodebook(p.num_nonterminals, p.stack_dim, p.codebook_initial_n, p.codebook_decay)
    elif codebook_type == "split":
        codebook = SplitCodebook(p.num_nonterminals, p.stack_dim, p.split_num, p.codebook_initial_n, p.codebook_decay)
    else:
        raise ValueError

    npda = NeuralPDA(
        pda_decoder=pda,

        nt_decoder=NTDecoder(
            rnn_cell=StackedRNNCell(
                RNNType=RNNType.VanillaRNN,
                input_dim=p.stack_dim,
                hidden_dim=p.stack_dim,
                n_layers=p.ntdec_layer,
                intermediate_dropout=p.dropout
            ),
            normalize_nt=p.ntdec_normalize,
        ),

        batch_stack=TensorBatchStack(max_batch_size=p.batch_sz,
                                     max_stack_size=150 if not hasattr(p, "stack_capacity") else p.stack_capacity,
                                     item_size=p.stack_dim),

        codebook=codebook,
        token_embedding=nn.Embedding(vocab.get_vocab_size(tgt_ns), p.token_dim),
        token_predictor=MoSProjection(5, p.token_dim, vocab.get_vocab_size(tgt_ns)),
        start_token_id=vocab.get_token_index(START_SYMBOL, tgt_ns),
        padding_token_id=vocab.get_token_index(PADDING_TOKEN, tgt_ns),

        ntdec_init_policy=p.ntdec_init,
        fn_decode_token=lambda i: vocab.get_token_from_index(i, namespace=tgt_ns),
    )

    model = NPDAFLM(npda, p.ntdec_factor)
    return model

def lm_ebnf(p, vocab: NSVocabulary):
    from models.neural_pda.ebnf_npda import NeuralEBNF
    from models.neural_pda.batched_stack import TensorBatchStack
    from models.neural_pda.formal_language_model import EBNFTreeLM
    from models.modules.stacked_rnn_cell import StackedLSTMCell
    from models.modules.quantized_token_predictor import QuantTokenPredictor

    emb_nt = nn.Embedding(vocab.get_vocab_size('nonterminal'), p.emb_sz, max_norm=p.nt_emb_max_norm)
    emb_t = nn.Embedding(vocab.get_vocab_size('terminal_category'), p.emb_sz, max_norm=p.t_emb_max_norm)

    nt_pred = QuantTokenPredictor(
        num_toks=vocab.get_vocab_size('nonterminal'),
        tok_dim=p.emb_sz,
        shared_embedding=emb_nt.weight if p.tied_nonterminal_emb else None,
        quant_criterion=p.nt_pred_crit,
    )
    t_pred = QuantTokenPredictor(
        num_toks=vocab.get_vocab_size('terminal_category'),
        tok_dim=p.emb_sz,
        shared_embedding=emb_t.weight if p.tied_terminal_emb else None,
        quant_criterion=p.t_pred_crit,
    )

    pda = NeuralEBNF(
        emb_nonterminals=emb_nt,
        emb_terminals=emb_t,
        num_nonterminals=vocab.get_token_index('nonterminal'),
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
        start_token_id=vocab.get_token_index(START_SYMBOL, 'terminal_category'),
        ebnf_entrypoint=vocab.get_token_index(p.grammar_entry, 'nonterminal'),
        padding_token_id=0,
        dropout=p.dropout,
    )

    model = EBNFTreeLM(pda)
    return model
