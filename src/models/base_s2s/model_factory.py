from typing import Union, Optional, Tuple, Dict, Any, Literal, List
import logging
import numpy as np
import torch
from torch import nn
from trialbot.data.ns_vocabulary import NSVocabulary
from ..interfaces.unified_rnn import UnifiedRNN
from ..interfaces.encoder import EncoderStack, EmbedAndEncode
from .base_seq2seq import BaseSeq2Seq
from .stacked_encoder import StackedEncoder, ExtLSTM
from models.transformer.encoder import TransformerEncoder
from ..modules.attention_wrapper import get_wrapped_attention
from ..modules.attention_composer import get_attn_composer
from .stacked_rnn_cell import StackedRNNCell
import os.path as osp


class EmbeddingMxin:
    def get_source_embedding(self):
        p, vocab = self.p, self.vocab
        src_pretrain_file = getattr(p, 'src_emb_pretrained_file', None)
        if src_pretrain_file is None:
            source_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.src_namespace),
                                            embedding_dim=p.emb_sz,
                                            padding_idx=0,
                                            )
        else:
            from allennlp.modules.token_embedders import Embedding
            source_embedding = Embedding(embedding_dim=p.emb_sz,
                                         num_embeddings=vocab.get_vocab_size(p.src_namespace),
                                         vocab_namespace=p.src_namespace,
                                         padding_index=0,
                                         pretrained_file=osp.expanduser(src_pretrain_file),
                                         vocab=vocab)
        return source_embedding

    def get_target_embedding(self):
        p, vocab = self.p, self.vocab
        target_embedding = nn.Embedding(vocab.get_vocab_size(p.tgt_namespace), p.emb_sz)
        return target_embedding

    def get_embeddings(self):
        p = self.p
        source_embedding = self.get_source_embedding()
        if p.src_namespace == p.tgt_namespace:
            target_embedding = source_embedding
        else:
            target_embedding = self.get_target_embedding()

        return source_embedding, target_embedding


class RNNListMixin:
    @staticmethod
    def get_stacked_rnns(cell_type: str, inp_sz: int, hid_sz: int, num_layers: int,
                         h_dropout: float, onlstm_chunk_sz: int = 10,
                         ) -> List[UnifiedRNN]:
        from ..modules.torch_rnn_wrapper import TorchRNNWrapper as RNNWrapper
        from ..modules.variational_dropout import VariationalDropout

        def _get_cell_in(floor): return inp_sz if floor == 0 else hid_sz
        def _get_h_vd(d): return VariationalDropout(d, on_the_fly=False) if d > 0 else None

        rnns = cls = None
        if cell_type == 'typed_rnn':
            from ..modules.sym_typed_rnn_cell import SymTypedRNNCell
            rnns = [SymTypedRNNCell(_get_cell_in(floor), hid_sz, "tanh", _get_h_vd(h_dropout))
                    for floor in range(num_layers)]

        elif cell_type == 'onlstm':
            from ..onlstm.onlstm import ONLSTMCell
            rnns = [ONLSTMCell(_get_cell_in(floor), hid_sz, onlstm_chunk_sz, _get_h_vd(h_dropout))
                    for floor in range(num_layers)]

        elif cell_type == 'ind_rnn':
            from ..modules.independent_rnn import IndRNNCell
            rnns = [IndRNNCell(_get_cell_in(floor), hid_sz) for floor in range(num_layers)]

        elif cell_type == "lstm":
            cls = torch.nn.LSTMCell
        elif cell_type == "gru":
            cls = torch.nn.GRUCell
        elif cell_type == "rnn":
            cls = torch.nn.RNNCell
        else:
            raise ValueError(f"RNN type of {cell_type} not found.")

        if rnns is None:
            rnns = [RNNWrapper(cls(_get_cell_in(floor), hid_sz), _get_h_vd(h_dropout))
                    for floor in range(num_layers)]

        return rnns


class EncoderStackMixin(RNNListMixin):
    @staticmethod
    def get_stacked_rnn_encoder(encoder_type: str, inp_sz, hid_sz, num_layers, dropout,
                                num_heads=12) -> EncoderStack:
        """
        p.enc_dropout = 0.
        p.enc_out_dim = xxx # otherwise p.hidden_sz is used
        p.emb_sz = 300
        p.encoder = "lstm"  # lstm, transformer, bilstm, aug_lstm, aug_bilstm
        p.num_heads = 8     # heads for transformer when used
        """
        from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper as PTRNNWrapper
        from allennlp.modules.seq2seq_encoders import AugmentedLstmSeq2SeqEncoder
        from allennlp.modules.seq2seq_encoders import StackedBidirectionalLstmSeq2SeqEncoder

        def _get_inp_sz(floor: int): return inp_sz if floor == 0 else hid_sz

        enc_classes = dict(
            lstm=lambda floor: PTRNNWrapper(nn.LSTM(_get_inp_sz(floor), hid_sz, batch_first=True)),
            gru=lambda floor: PTRNNWrapper(nn.GRU(_get_inp_sz(floor), hid_sz, batch_first=True)),
            bilstm=lambda floor: PTRNNWrapper(nn.LSTM(_get_inp_sz(floor), hid_sz, batch_first=True,
                                                      bidirectional=True)),
            bigru=lambda floor: PTRNNWrapper(nn.GRU(_get_inp_sz(floor), hid_sz, batch_first=True,
                                                    bidirectional=True)),
            torch_bilstm=lambda floor: ExtLSTM(nn.LSTM(_get_inp_sz(floor), hid_sz, batch_first=True,
                                                       bidirectional=True)),
            aug_lstm=lambda floor: AugmentedLstmSeq2SeqEncoder(
                _get_inp_sz(floor), hid_sz,
                recurrent_dropout_probability=dropout, use_highway=True,
            ),
            aug_bilstm=lambda floor: StackedBidirectionalLstmSeq2SeqEncoder(
                _get_inp_sz(floor), hid_sz, num_layers=1,
                recurrent_dropout_probability=dropout, use_highway=True,
            ),
            transformer=lambda floor: TransformerEncoder(
                _get_inp_sz(floor), hid_sz,
                num_layers=1,
                num_heads=num_heads,
                feedforward_hidden_dim=hid_sz,
                feedforward_dropout=dropout,
                residual_dropout=dropout,
                attention_dropout=0.,
                use_positional_embedding=(floor == 0),
            ),
        )

        enc_cls = enc_classes.get(encoder_type)
        assert enc_cls is not None
        encoder = StackedEncoder([enc_cls(floor) for floor in range(num_layers)],
                                 input_size=inp_sz,
                                 output_size=hid_sz,
                                 input_dropout=dropout)
        return encoder

    @staticmethod
    def get_diora_encoder(encoder_type: str, inp_sz, hid_sz, concat_outside: bool = False) -> EncoderStack:
        from ..diora.hard_diora import DioraTopk
        from ..diora.diora import Diora
        from ..diora.diora_encoder import DioraEncoder

        assert encoder_type in ('diora', 's-diora'), f'unsupported diora type {encoder_type}'

        diora_cls = {
            'diora': Diora,
            's-diora': DioraTopk,
        }.get(encoder_type)

        diora = diora_cls.from_kwargs_dict(dict(size=hid_sz, input_size=inp_sz))
        enc = DioraEncoder(diora, concat_outside)
        return enc

    @staticmethod
    def get_perturb_parse_gcn_encoder(inp_sz, hid_sz, num_layers) -> EncoderStack:
        from ..perturb_and_parse.pp_encoder import PerturbParseEncoder
        enc = PerturbParseEncoder(inp_sz, hid_sz, num_layers)
        return enc

    def get_stacked_cell_encoder(self) -> EncoderStack:
        p = self.p
        dropout = getattr(p, 'enc_dropout', getattr(p, 'dropout', 0.))
        hid_sz = getattr(p, 'enc_out_dim', p.hidden_sz)
        bid_cell = getattr(p, 'cell_encoder_is_bidirectional', False)
        use_pseq = getattr(p, 'cell_encoder_uses_packed_sequence', False)

        rnns = self.get_stacked_rnns(p.encoder, p.emb_sz, hid_sz, p.num_enc_layers, dropout)
        b_rnns = self.get_stacked_rnns(p.encoder, p.emb_sz, hid_sz, p.num_enc_layers, dropout)

        from .cell_encoder import CellEncoder
        return StackedEncoder([CellEncoder(rnn, brnn, use_pseq)
                               if bid_cell else
                               CellEncoder(rnn, None, use_pseq)
                               for rnn, brnn in zip(rnns, b_rnns)],
                              input_dropout=dropout)

    def get_encoder_stack(self) -> EncoderStack:
        p = self.p
        dropout = getattr(p, 'enc_dropout', getattr(p, 'dropout', 0.))
        hid_sz = getattr(p, 'enc_out_dim', p.hidden_sz)
        num_heads = getattr(p, 'num_heads', 16 if hid_sz % 16 == 0 else 10)

        if 'diora' in p.encoder:
            concat = getattr(p, 'diora_concat_outside', False)
            return self.get_diora_encoder(p.encoder, p.emb_sz, hid_sz, concat)

        if 'perturb_parse' == p.encoder:
            return self.get_perturb_parse_gcn_encoder(p.emb_sz, hid_sz, p.num_enc_layers)

        if getattr(p, 'use_cell_based_encoder', False):
            return self.get_stacked_cell_encoder()

        return self.get_stacked_rnn_encoder(p.encoder, p.emb_sz, hid_sz, p.num_enc_layers, dropout, num_heads)


class EmbEncBundleMixin:
    def get_embed_encoder_bundle(self, emb, enc: EncoderStack, padding_idx) -> EmbedAndEncode:
        p, vocab = self.p, self.vocab
        enc_dropout = getattr(p, 'enc_dropout', getattr(p, 'dropout', 0.))
        if p.encoder == 'diora':
            from ..diora.diora_bundle import SeqEmbedAndDiora
            return SeqEmbedAndDiora(emb, enc, padding_idx, enc_dropout,
                                    use_diora_loss=getattr(p, 'diora_loss_enabled', False))

        from .seq_embed_encode import SeqEmbedEncoder
        emb_enc = SeqEmbedEncoder(emb, enc, padding_idx, enc_dropout)
        emb_enc = self.get_seq_compound_bundle(emb_enc)
        return emb_enc

    def get_seq_compound_bundle(self, emb_enc: EmbedAndEncode):
        p, vocab = self.p, self.vocab
        # a compound EmbEnc is an EmbEnc that depends on another EmbEnc.
        compound_emb_enc = getattr(p, 'compound_encoder', None)

        if compound_emb_enc == 'cpcfg':
            from ..pcfg.pcfg_emb_enc import CompoundPCFGEmbedEncode
            from ..pcfg.C_PCFG import CompoundPCFG
            return CompoundPCFGEmbedEncode(
                pcfg=CompoundPCFG(
                    num_nonterminal=p.num_pcfg_nt,
                    num_preterminal=p.num_pcfg_pt,
                    num_vocab_token=vocab.get_vocab_size(p.src_namespace),
                    hidden_sz=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                    z_dim=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                    encoder_input_dim=emb_enc.get_output_dim(),
                    emb_chart_dim=getattr(p, 'pcfg_encoding_dim', p.hidden_sz),
                ),
                emb_enc=emb_enc,
                z_dim=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
            )
        elif compound_emb_enc == 'reduced_cpcfg':
            from ..pcfg.pcfg_emb_enc import CompoundPCFGEmbedEncode
            from ..pcfg.reduced_c_pcfg import ReducedCPCFG
            return CompoundPCFGEmbedEncode(
                pcfg=ReducedCPCFG(
                    num_nonterminal=p.num_pcfg_nt,
                    num_preterminal=p.num_pcfg_pt,
                    num_vocab_token=vocab.get_vocab_size(p.src_namespace),
                    hidden_sz=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                    z_dim=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                    encoder_input_dim=emb_enc.get_output_dim(),
                    emb_chart_dim=getattr(p, 'pcfg_encoding_dim', p.hidden_sz),
                ),
                emb_enc=emb_enc,
                z_dim=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
            )
        elif compound_emb_enc == 'tdpcfg':
            from ..pcfg.pcfg_emb_enc import CompoundPCFGEmbedEncode
            from ..pcfg.TN_PCFG import TNPCFG
            return CompoundPCFGEmbedEncode(
                pcfg=TNPCFG(
                    rank=getattr(p, 'td_pcfg_rank', p.num_pcfg_nt // 10),
                    num_nonterminal=p.num_pcfg_nt,
                    num_preterminal=p.num_pcfg_pt,
                    num_vocab_token=vocab.get_vocab_size(p.src_namespace),
                    hidden_sz=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                    z_dim=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                    encoder_input_dim=emb_enc.get_output_dim(),
                    emb_chart_dim=getattr(p, 'pcfg_encoding_dim', p.hidden_sz),
                ),
                emb_enc=emb_enc,
                z_dim=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
            )
        elif compound_emb_enc == 'reduced_tdpcfg':
            from ..pcfg.pcfg_emb_enc import CompoundPCFGEmbedEncode
            from ..pcfg.reduced_td_pcfg import ReducedTDPCFG
            return CompoundPCFGEmbedEncode(
                pcfg=ReducedTDPCFG(
                    rank=getattr(p, 'td_pcfg_rank', p.num_pcfg_nt // 10),
                    num_nonterminal=p.num_pcfg_nt,
                    num_preterminal=p.num_pcfg_pt,
                    num_vocab_token=vocab.get_vocab_size(p.src_namespace),
                    hidden_sz=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                    z_dim=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                    encoder_input_dim=emb_enc.get_output_dim(),
                    emb_chart_dim=getattr(p, 'pcfg_encoding_dim', p.hidden_sz),
                ),
                emb_enc=emb_enc,
                z_dim=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
            )

        return emb_enc


class WordProjMixin:
    def get_word_projection(self, proj_in_dim, target_embedding=None):
        p, vocab = self.p, self.vocab
        word_proj = nn.Linear(proj_in_dim, vocab.get_vocab_size(p.tgt_namespace))
        tied_emb = target_embedding is not None and p.tied_decoder_embedding
        logging.getLogger(self.__class__.__name__).info(f"Decoding {'' if tied_emb else 'NOT'} using tied embeddings")
        if tied_emb:
            emb_sz = target_embedding.weight.size()[1]
            assert proj_in_dim == emb_sz, f"Tied embeddings must have the same dimensions, proj{proj_in_dim} != emb{emb_sz}"
            word_proj.weight = target_embedding.weight  # tied embedding
        return word_proj


class Seq2SeqBuilder(EmbeddingMxin,
                     EncoderStackMixin,
                     EmbEncBundleMixin,
                     WordProjMixin,
                     ):
    """
    Build the Seq2Seq model. Typical parts include:
    Model (
        EmbedAndEncodeBundle (
            Source Embedding,
            EncoderStack (      // e.g. interface, implemented by StackedEncoder or DioraEncoder
                [Encoder], or   // e.g. interface, the nn.LSTM or CellEncoder wrapped by StackedEncoder
                DioraBase, or   // e.g. base class of Diora and DioraTopk, used in DioraEncoder
                ...
            )
        )
        Target Embedding,
        RNNStack ( [UnifiedRNN], ),
        WordProjection,

        EncDecTransformation Module,

        Attention for Encoder,
        Attention for Decoder History,
        Attention Composer for Decoder Input,
        Attention Composer for Projection Input,
    )

    Build the seq2seq model with hyper-parameters.
    Since S2S is widely used and can be implemented task-agnostic,
    the builder for trialbot is provided as default.
    Because we want a flat configuration file and use it only with the BaseSeq2Seq model,
    the model is responsible for providing definitions.

    p.emb_sz = 256
    p.src_namespace = 'ns_q'
    p.tgt_namespace = 'ns_lf'
    p.hidden_sz = 128
    p.dec_in_dim = p.hidden_sz # by default
    p.dec_out_dim = p.hidden_sz # by default
    p.proj_in_dim = p.hidden_sz # by default
    p.enc_attn = "bilinear"
    p.dec_hist_attn = "dot_product"
    p.dec_inp_composer = 'cat_mapping'
    p.dec_inp_comp_activation = 'mish'
    p.proj_inp_composer = 'cat_mapping'
    p.proj_inp_comp_activation = 'mish'
    p.enc_dec_trans_act = 'linear'
    p.enc_dec_trans_usage = 'consistent'
    p.enc_dec_trans_forced = True
    p.use_cell_based_encoder = False
    p.encoder = "bilstm"
    p.cell_encoder_is_bidirectional = False
    p.num_enc_layers = 2
    p.decoder = "lstm"
    p.num_dec_layers = 2
    p.dropout = .2
    p.enc_dropout = p.dropout # by default
    p.dec_dropout = p.dropout # by default
    p.max_decoding_step = 100
    p.scheduled_sampling = .1
    p.decoder_init_strategy = "forward_last_parallel"
    p.tied_decoder_embedding = True
    p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"
    """

    def __init__(self, p, vocab):
        super().__init__()
        self.p = p
        self.vocab = vocab

    def get_model(self) -> BaseSeq2Seq:
        from trialbot.data import START_SYMBOL, END_SYMBOL

        p, vocab = self.p, self.vocab
        emb_sz = p.emb_sz

        source_embedding, target_embedding = self.get_embeddings()
        encoder = self.get_encoder_stack()
        embed_and_encoder = self.get_embed_encoder_bundle(source_embedding, encoder, padding_idx=0)

        enc_out_dim = embed_and_encoder.get_output_dim()
        dec_in_dim = getattr(p, 'dec_in_dim', p.hidden_sz)
        dec_out_dim = getattr(p, 'dec_out_dim', p.hidden_sz)
        proj_in_dim = getattr(p, 'proj_in_dim', p.hidden_sz)

        trans_module, _usages = self._get_transformation_module(encoder.is_bidirectional(), enc_out_dim, dec_out_dim)
        trans_usage_string = self._get_trans_usage_string(trans_module, _usages)

        # Initialize attentions. And compute the dimension requirements for all attention modules
        # the encoder attention size depends on the transformation usage
        enc_attn_sz = dec_out_dim if trans_usage_string in ('consistent', 'attn') else enc_out_dim
        enc_attn = get_wrapped_attention(p.enc_attn, dec_out_dim, enc_attn_sz)
        # dec output attend over previous dec outputs, thus attn_context dimension == dec_output_dim
        dec_hist_attn = get_wrapped_attention(p.dec_hist_attn, dec_out_dim, dec_out_dim)
        attn_sz = 0 if dec_hist_attn is None else dec_out_dim
        attn_sz += 0 if enc_attn is None else enc_attn_sz
        dec_inp_composer = get_attn_composer(p.dec_inp_composer, attn_sz, emb_sz, dec_in_dim, p.dec_inp_comp_activation)
        proj_inp_composer = get_attn_composer(p.proj_inp_composer, attn_sz, dec_out_dim, proj_in_dim, p.proj_inp_comp_activation)

        if enc_attn is not None or dec_hist_attn is not None:
            assert dec_inp_composer is not None and proj_inp_composer is not None, "Attention must be composed"

        dec_dropout = getattr(p, 'dec_dropout', getattr(p, 'dropout', 0.))

        rnns = self.get_stacked_rnns(p.decoder, dec_in_dim, dec_out_dim, p.num_dec_layers, dec_dropout)
        decoder = StackedRNNCell(rnns, dec_dropout)

        word_proj = self.get_word_projection(proj_in_dim, target_embedding)

        model = BaseSeq2Seq(
            vocab=vocab,
            embed_encoder=embed_and_encoder,
            decoder=decoder,
            word_projection=word_proj,
            target_embedding=target_embedding,
            enc_attention=enc_attn,
            dec_hist_attn=dec_hist_attn,
            dec_inp_attn_comp=dec_inp_composer,
            proj_inp_attn_comp=proj_inp_composer,
            enc_dec_transformer=trans_module,
            source_namespace=p.src_namespace,
            target_namespace=p.tgt_namespace,
            start_symbol=START_SYMBOL,
            eos_symbol=END_SYMBOL,
            padding_index=0,
            max_decoding_step=p.max_decoding_step,
            decoder_init_strategy=p.decoder_init_strategy,
            enc_dec_transform_usage=trans_usage_string,
            scheduled_sampling_ratio=p.scheduled_sampling,
            dec_dropout=dec_dropout,
            training_average=getattr(p, "training_average", "batch"),
        )
        return model

    def _get_transformation_module(self, enc_is_bidirectional: bool, enc_out_dim: int, dec_out_dim: int):
        import allennlp.nn
        p = self.p
        # autodetect for the necessity of the transformation module
        # expecting the dimensions matchs between the decoder and encoder, otherwise a transformer is introduced.
        trans_for_dec_init = not (p.decoder_init_strategy.startswith('zero')
                                  or (enc_is_bidirectional and dec_out_dim * 2 == enc_out_dim)
                                  or (not enc_is_bidirectional and dec_out_dim == enc_out_dim))
        # expecting the encoder output and decoder output matches when attention is dot-product,
        # other attention types will transform the output dimension on their own.
        trans_for_attn = (p.enc_attn == 'dot_product' and enc_out_dim != dec_out_dim)
        # the transformer is configured to be
        forced_transformer: bool = getattr(p, 'enc_dec_trans_forced', False)
        enc_dec_transformer = None
        if trans_for_attn or trans_for_dec_init or forced_transformer:
            enc_dec_transformer = nn.Sequential(
                nn.Linear(enc_out_dim, dec_out_dim),
                allennlp.nn.Activation.by_name(getattr(p, 'enc_dec_trans_act', 'linear'))(),
            )
        return enc_dec_transformer, (trans_for_dec_init, trans_for_attn)

    def _get_trans_usage_string(self, trans_module, usage):
        p = self.p
        used_by_dec_init, used_by_attn = usage
        # when transformer is present, check if the consistent transformation is preferred.
        # if it is so, a mere dec-init or attn usage will get overwritten
        # consistent will not work if the transformer is unavailable
        preferred_consistent: bool = getattr(p, 'enc_dec_trans_usage', 'consistent').lower() == 'consistent'
        if preferred_consistent and trans_module is not None:
            enc_dec_trans_usage = "consistent"  # the output will be transformed immediately and never in the future
        elif used_by_dec_init:
            enc_dec_trans_usage = "dec_init"    # the output will be only transformed when initializing the decoder
        elif used_by_attn:
            enc_dec_trans_usage = "attn"    # the output will be only transformed when computing attentions
        else:
            enc_dec_trans_usage = ""    # no need at all

        return enc_dec_trans_usage

    @classmethod
    def from_param_and_vocab(cls, p, vocab: NSVocabulary):
        return Seq2SeqBuilder(p, vocab).get_model()

