import training.exp_runner
import allennlp.modules

from allennlp.data import Vocabulary
import config

import torch.nn as nn
from models.transformer.encoder import TransformerEncoder
from models.transformer.multi_head_attention import GeneralMultiHeadAttention, SingleTokenMHAttentionWrapper
from models.modules.stacked_encoder import StackedEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from models.modules.universal_hidden_state_wrapper import UniversalHiddenStateWrapper, RNNType
from models.modules.stacked_rnn_cell import StackedLSTMCell, StackedGRUCell
from allennlp.modules.attention import BilinearAttention, DotProductAttention
from utils.nn import AllenNLPAttentionWrapper
from models.base_s2s.base_seq2seq import BaseSeq2Seq
from allennlp.common.util import START_SYMBOL, END_SYMBOL


@config.register_hparams
def trans2seq_mspars_hparams():
    hparams = config.common_settings()
    hparams.batch_sz = 128
    hparams.emb_sz = 256
    hparams.encoder = 'transformer'
    hparams.num_enc_layers = 1
    hparams.attention_dropout = 0
    hparams.intermediate_dropout = 0.2
    hparams.max_decoding_len = 15
    hparams.dec_cell_height = 1

    hparams.mha_attn_dim = hparams.emb_sz
    hparams.mha_value_dim = hparams.emb_sz
    hparams.enc_attn = "bilinear"
    hparams.dec_hist_attn = "none"
    hparams.concat_attn_to_dec_input = False

    hparams.tied_decoder_embedding = False
    return hparams

def get_model(hparams, vocab: Vocabulary):
    emb_sz = hparams.emb_sz

    source_embedding = allennlp.modules.Embedding(
        num_embeddings=vocab.get_vocab_size('src_tokens'),
        embedding_dim=emb_sz,
    )
    target_embedding = allennlp.modules.Embedding(
        num_embeddings=vocab.get_vocab_size('tgt_tokens'),
        embedding_dim=emb_sz,
    )

    encoder = get_encoder(hparams)
    enc_out_dim = encoder.get_output_dim()
    dec_out_dim = emb_sz
    dec_hist_attn = get_attention(hparams, emb_sz, dec_out_dim, hparams.dec_hist_attn)
    enc_attn = get_attention(hparams, emb_sz, enc_out_dim, hparams.enc_attn)

    def sum_attn_dims(attns, dims):
        return sum(dim for attn, dim in zip(attns, dims) if attn is not None)

    # all the attention vectors are concatenated with input
    dec_in_dim = dec_out_dim + sum_attn_dims([enc_attn, dec_hist_attn], [enc_out_dim, dec_out_dim])
    rnn_cell = get_rnn_cell(hparams, dec_in_dim, dec_out_dim)

    # tied decoder weights connect the vocab projection and input embedding together.
    proj_in_dim = dec_out_dim + sum_attn_dims([enc_attn, dec_hist_attn], [enc_out_dim, dec_out_dim])
    if hparams.tied_decoder_embedding:
        word_proj = nn.Sequential(
            nn.Linear(proj_in_dim, emb_sz),
            nn.Linear(emb_sz, vocab.get_vocab_size('tgt_tokens')),
        )
    else:
        word_proj = nn.Linear(proj_in_dim, vocab.get_vocab_size('tgt_tokens'))

    model = BaseSeq2Seq(vocab=vocab,
                        encoder=encoder,
                        decoder=rnn_cell,
                        word_projection=word_proj,
                        source_embedding=source_embedding,
                        target_embedding=target_embedding,
                        target_namespace='tgt_tokens',
                        start_symbol=START_SYMBOL,
                        eos_symbol=END_SYMBOL,
                        max_decoding_step=hparams.max_decoding_len,
                        enc_attention=enc_attn,
                        dec_hist_attn=dec_hist_attn,
                        intermediate_dropout=hparams.intermediate_dropout,
                        concat_attn_to_dec_input=hparams.concat_attn_to_dec_input,
                        )

    return model

def get_encoder(hparams):
    emb_sz = hparams.emb_sz
    if hparams.encoder == 'lstm':
        encoder = StackedEncoder([
            PytorchSeq2SeqWrapper(nn.LSTM(emb_sz, emb_sz, batch_first=True))
            for _ in range(hparams.num_enc_layers)
        ], emb_sz, emb_sz, input_dropout=hparams.intermediate_dropout)
    elif hparams.encoder == 'bilstm':
        encoder = StackedEncoder([
            PytorchSeq2SeqWrapper(nn.LSTM(emb_sz, emb_sz, batch_first=True, bidirectional=True))
            for _ in range(hparams.num_enc_layers)
        ], emb_sz, emb_sz, input_dropout=hparams.intermediate_dropout)
    elif hparams.encoder == 'transformer':
        encoder = StackedEncoder([
            TransformerEncoder(input_dim=emb_sz,
                               num_layers=1,
                               num_heads=hparams.num_heads,
                               feedforward_hidden_dim=emb_sz,
                               feedforward_dropout=hparams.feedforward_dropout,
                               residual_dropout=hparams.residual_dropout,
                               attention_dropout=hparams.attention_dropout,
                               )
            for _ in range(hparams.num_enc_layers)
        ], emb_sz, emb_sz, input_dropout=hparams.intermediate_dropout)
    else:
        assert False

    return encoder

def get_rnn_cell(hparams, input_dim: int, hidden_dim: int):
    cell_type = hparams.decoder
    if cell_type == "lstm":
        return UniversalHiddenStateWrapper(RNNType.LSTM(input_dim, hidden_dim))
    elif cell_type == "gru":
        return UniversalHiddenStateWrapper(RNNType.GRU(input_dim, hidden_dim))
    elif cell_type == "ind_rnn":
        return UniversalHiddenStateWrapper(RNNType.IndRNN(input_dim, hidden_dim))
    elif cell_type == "rnn":
        return UniversalHiddenStateWrapper(RNNType.VanillaRNN(input_dim, hidden_dim))
    elif cell_type == 'n_lstm':
        n_layer = hparams.dec_cell_height
        return StackedLSTMCell(input_dim, hidden_dim, n_layer, hparams.intermediate_dropout)
    elif cell_type == 'n_gru':
        n_layer = hparams.dec_cell_height
        return StackedGRUCell(input_dim, hidden_dim, n_layer, hparams.intermediate_dropout)
    else:
        raise ValueError(f"RNN type of {cell_type} not found.")

def get_attention(hparams, vec_emb: int, attend_to_emb: int, attn_type: str):
    attn_type = attn_type.lower()
    if attn_type == "bilinear":
        attn = BilinearAttention(vector_dim=vec_emb, matrix_dim=attend_to_emb)
        attn = AllenNLPAttentionWrapper(attn, hparams.attention_dropout)

    elif attn_type == "dot_product":
        assert vec_emb == attend_to_emb
        attn = DotProductAttention()
        attn = AllenNLPAttentionWrapper(attn, hparams.attention_dropout)

    elif attn_type == "multihead":
        attn = GeneralMultiHeadAttention(num_heads=hparams.num_heads,
                                         input_dim=vec_emb,
                                         total_attention_dim=hparams.mha_attn_dim,
                                         total_value_dim=hparams.mha_value_dim,
                                         attend_to_dim=attend_to_emb,
                                         output_dim=attend_to_emb,
                                         attention_dropout=hparams.attention_dropout,
                                         use_future_blinding=False,
                                         )
        attn = SingleTokenMHAttentionWrapper(attn)
    elif attn_type == "none":
        attn = None
    else:
        assert False

    return attn

if __name__ == '__main__':
    import data_adapter
    runner = training.exp_runner.ExperimentRunner(exp_name='s2s')
    # runner.reader = data_adapter.DATA_READERS[runner.args.dataset]()
    runner.run()


