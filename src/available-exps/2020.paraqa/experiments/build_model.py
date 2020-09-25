from trialbot.data import NSVocabulary, PADDING_TOKEN, START_SYMBOL, END_SYMBOL
from torch import nn

def get_seq2seq_model(p, vocab: NSVocabulary):
    from models.base_s2s.base_seq2seq import BaseSeq2Seq
    from models.modules.stacked_rnn_cell import StackedLSTMCell, StackedGRUCell, StackedRNNCell

    emb_sz = p.emb_sz
    source_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.src_namespace), embedding_dim=emb_sz)
    target_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.tgt_namespace), embedding_dim=emb_sz)

    encoder = _get_encoder(p)
    enc_out_dim = encoder.get_output_dim()
    dec_out_dim = p.hidden_sz    # the output dimension of decoder is just the same as hidden_dim

    # dec output attend over previous dec outputs, thus attn_context dimension == dec_output_dim
    dec_hist_attn = _get_attention(p.dec_hist_attn, dec_out_dim, dec_out_dim)
    # dec output attend over encoder outputs, thus attn_context dimension == enc_output_dim
    enc_attn = _get_attention(p.enc_attn, dec_out_dim, enc_out_dim)

    if p.enc_attn == 'dot_product':
        assert enc_out_dim == dec_out_dim, "encoder hidden states must be able to multiply with decoder output"

    def sum_attn_dims(attns, dims):
        """Compute the dimension requirements for all attention modules"""
        return sum(dim for attn, dim in zip(attns, dims) if attn is not None)
    # since attentions are mapped into hidden size, the hidden_dim is used instead of original context size
    attn_size = sum_attn_dims([enc_attn, dec_hist_attn], [dec_out_dim, dec_out_dim])

    dec_in_dim = emb_sz + (attn_size if p.concat_attn_to_dec_input else 0)
    rnn_cls = _get_decoder_type(p)
    decoder = StackedRNNCell(rnn_cls, dec_in_dim, dec_out_dim, p.num_dec_layers, p.dropout)

    proj_in_dim = dec_out_dim + (attn_size if p.concat_attn_to_dec_input else 0)
    word_proj = nn.Linear(proj_in_dim, vocab.get_vocab_size(p.tgt_namespace))

    model = BaseSeq2Seq(
        vocab=vocab,
        encoder=encoder,
        decoder=decoder,
        source_embedding=source_embedding,
        target_embedding=target_embedding,
        word_projection=word_proj,
        target_namespace=p.tgt_namespace,
        start_symbol=START_SYMBOL,
        eos_symbol=END_SYMBOL,
        max_decoding_step=p.max_decoding_step,
        enc_attention=enc_attn,
        dec_hist_attn=dec_hist_attn,
        scheduled_sampling_ratio=p.scheduled_sampling,
        intermediate_dropout=p.dropout,
        concat_attn_to_dec_input=p.concat_attn_to_dec_input,
        padding_index=0,
        decoder_init_strategy=p.decoder_init_strategy,
    )
    return model

def _get_encoder(p):
    from models.modules.stacked_encoder import StackedEncoder
    from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
    from models.transformer.encoder import TransformerEncoder
    emb_sz, hidden_dim = p.emb_sz, p.hidden_sz

    if p.encoder == 'lstm':
        encoder = StackedEncoder([
            PytorchSeq2SeqWrapper(nn.LSTM(emb_sz if floor == 0 else hidden_dim, hidden_dim, batch_first=True))
            for floor in range(p.num_enc_layers)
        ], emb_sz, hidden_dim, input_dropout=p.dropout)
    elif p.encoder == 'bilstm':
        encoder = StackedEncoder([
            PytorchSeq2SeqWrapper(nn.LSTM(emb_sz if floor == 0 else hidden_dim * 2,
                                          hidden_dim, batch_first=True, bidirectional=True))
            for floor in range(p.num_enc_layers)
        ], emb_sz, hidden_dim, input_dropout=p.dropout)
    elif p.encoder == 'transformer':
        encoder = StackedEncoder([
            TransformerEncoder(input_dim=emb_sz if floor == 0 else hidden_dim,
                               hidden_dim=hidden_dim,
                               num_layers=1,
                               num_heads=p.num_heads,
                               feedforward_hidden_dim=hidden_dim,
                               feedforward_dropout=p.dropout,
                               residual_dropout=p.dropout,
                               attention_dropout=p.attention_dropout,
                               use_positional_embedding=(floor == 0)
                               )
            for floor in range(p.num_enc_layers)
        ], emb_sz, hidden_dim, input_dropout=p.dropout)
    else:
        assert False

    return encoder

def _get_attention(attn_type: str, vector_dim: int, matrix_dim: int, attention_dropout: float = 0., **kwargs):
    """
    Build an Attention module with specified parameters.
    :param attn_type: indicates the attention type, e.g. "bilinear", "dot_product" or "none"
    :param vector_dim: the vector to compute attention
    :param matrix_dim: the bunch of vectors to attend against (batch, num, matrix_dim)
    :param attention_dropout: the dropout to discard some attention weights
    :return: a torch.nn.Module
    """
    from utils.nn import AllenNLPAttentionWrapper
    from allennlp.modules.attention import BilinearAttention, DotProductAttention

    attn_type = attn_type.lower()
    if attn_type == "bilinear":
        attn = BilinearAttention(vector_dim=vector_dim, matrix_dim=matrix_dim)
        attn = AllenNLPAttentionWrapper(attn, attention_dropout)
    elif attn_type == "dot_product":
        attn = DotProductAttention()
        attn = AllenNLPAttentionWrapper(attn, attention_dropout)
    elif attn_type == "none":
        attn = None
    else:
        assert False

    return attn

def _get_decoder_type(p):
    from models.modules.universal_hidden_state_wrapper import UniversalHiddenStateWrapper, RNNType
    cell_type = p.decoder
    if cell_type == "lstm":
        cls = RNNType.LSTM
    elif cell_type == "gru":
        cls = RNNType.GRU
    elif cell_type == "ind_rnn":
        cls = RNNType.IndRNN
    elif cell_type == "rnn":
        cls = RNNType.VanillaRNN
    else:
        raise ValueError(f"RNN type of {cell_type} not found.")

    return cls
