from trialbot.data import NSVocabulary, PADDING_TOKEN, START_SYMBOL, END_SYMBOL
from torch import nn

def lm_npda(p, vocab: NSVocabulary):
    from models.neural_pda.formal_language_model import NPDAFLM
    from models.neural_pda.npda import NeuralPDA, NTDecoder
    from models.neural_pda.batched_stack import BatchStack, TensorBatchStack, ListedBatchStack
    from models.neural_pda.npda_cell import TRNNPDACell, LSTMPDACell
    from models.modules.stacked_rnn_cell import StackedRNNCell, RNNType
    from models.modules.mixture_softmax import MoSProjection
    from models.neural_pda.codebook import CodeBook
    tgt_ns = getattr(p, 'target_namespace', 'sparqlPattern')

    pda_type = getattr(p, 'pda_type', 'trnn')
    if pda_type == 'lstm':
        pda = LSTMPDACell(p.token_dim, p.stack_dim, p.hidden_dim)
    else:
        pda = TRNNPDACell(p.token_dim, p.stack_dim, p.hidden_dim)

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

        codebook=CodeBook(num_nonterminals=p.num_nonterminals,
                          nonterminal_dim=p.stack_dim,
                          init_codebook_confidence=p.codebook_initial_n,
                          codebook_training_decay=p.codebook_decay),

        token_embedding=nn.Embedding(vocab.get_vocab_size(tgt_ns), p.token_dim),
        token_predictor=MoSProjection(5, p.token_dim, vocab.get_vocab_size(tgt_ns)),
        start_token_id=vocab.get_token_index(START_SYMBOL, tgt_ns),
        padding_token_id=vocab.get_token_index(PADDING_TOKEN, tgt_ns),

        ntdec_init_policy=p.ntdec_init,
        fn_decode_token=lambda i: vocab.get_token_from_index(i, namespace=tgt_ns),
    )

    model = NPDAFLM(npda, p.ntdec_factor)
    return model
