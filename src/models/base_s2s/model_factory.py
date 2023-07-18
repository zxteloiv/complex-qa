import logging
import torch
from torch import nn
from trialbot.data.ns_vocabulary import NSVocabulary
from ..interfaces.unified_rnn import UnifiedRNN
from ..interfaces.encoder import StackEncoder, EmbedAndEncode, EmbedAndGraphEncode
from .base_seq2seq import BaseSeq2Seq
from .syngraph2seq import SynGraph2Seq
from .encoder_stacker import EncoderStacker, ExtLSTM
from models.transformer.encoder import TransformerEncoder
from ..modules.attentions import get_attn_composer, get_attention
from .rnn_stacker import RNNCellStacker
import os.path as osp
import numpy
import tqdm


logger = logging.getLogger(__name__)


class EmbeddingMixin:
    def get_source_embedding(self):
        p, vocab = self.p, self.vocab
        if p.src_namespace is None:
            return None

        src_pretrain_file = getattr(p, 'src_emb_pretrained_file', None)
        if src_pretrain_file is None:
            source_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.src_namespace),
                                            embedding_dim=p.emb_sz,
                                            padding_idx=0)
        else:
            weights = self._read_pretrained_embeddings_file(
                osp.expanduser(src_pretrain_file), p.emb_sz, vocab, p.src_namespace,
            )
            source_embedding = nn.Embedding(num_embeddings=vocab.get_vocab_size(p.src_namespace),
                                            embedding_dim=p.emb_sz,
                                            padding_idx=0,
                                            _weight=weights)
        return source_embedding

    @staticmethod
    def _read_pretrained_embeddings_file(
        filename: str, embedding_dim: int, vocab: NSVocabulary, namespace: str = "tokens"
    ) -> torch.FloatTensor:
        """
        Copied and modified from the original AllenNLP code base.

        Read pre-trained word vectors from an eventually compressed text file, possibly contained
        inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
        space-separated fields: [word] [dim 1] [dim 2] ...

        Lines that contain more numerical tokens than `embedding_dim` raise a warning and are skipped.

        The remainder of the docstring is identical to `_read_pretrained_embeddings_file`.
        """
        tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
        vocab_size = vocab.get_vocab_size(namespace)
        embeddings = {}

        # First we read the embeddings from the file, only keeping vectors for the words we need.
        logger.info(f"Reading pretrained embeddings from file {filename}")
        from trialbot.utils.file_reader import open_file

        with open_file(filename) as embeddings_file:
            for line in tqdm.tqdm(embeddings_file):
                token = line.split(" ", 1)[0]
                if token in tokens_to_keep:
                    fields = line.rstrip().split(" ")
                    if len(fields) - 1 != embedding_dim:
                        # Sometimes there are funny unicode parsing problems that lead to different
                        # fields lengths (e.g., a word with a unicode space character that splits
                        # into more than one column).  We skip those lines.  Note that if you have
                        # some kind of long header, this could result in all of your lines getting
                        # skipped.  It's hard to check for that here; you just have to look in the
                        # embedding_misses_file and at the model summary to make sure things look
                        # like they are supposed to.
                        logger.warning(
                            "Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                            embedding_dim,
                            len(fields) - 1,
                            line,
                            )
                        continue

                    vector = numpy.asarray(fields[1:], dtype="float32")
                    embeddings[token] = vector

        if not embeddings:
            raise ValueError(
                "No embeddings of correct dimension found; you probably "
                "misspecified your embedding_dim parameter, or didn't "
                "pre-populate your Vocabulary"
            )

        all_embeddings = numpy.asarray(list(embeddings.values()))
        embeddings_mean = float(numpy.mean(all_embeddings))
        embeddings_std = float(numpy.std(all_embeddings))
        # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
        # then filling in the word vectors we just read.
        logger.info("Initializing pre-trained embedding layer")
        embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(
            embeddings_mean, embeddings_std
        )
        num_tokens_found = 0
        index_to_token = vocab.get_index_to_token_vocabulary(namespace)
        for i in range(vocab_size):
            token = index_to_token[i]

            # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
            # so the word has a random initialization.
            if token in embeddings:
                embedding_matrix[i] = torch.FloatTensor(embeddings[token])
                num_tokens_found += 1
            else:
                logger.debug(
                    "Token %s was not found in the embedding file. Initialising randomly.", token
                )

        logger.info(
            "Pretrained embeddings were found for %d out of %d tokens", num_tokens_found, vocab_size
        )

        return embedding_matrix

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
    def get_rnn_list(cell_type: str, inp_sz: int, hid_sz: int, num_layers: int,
                     h_dropout: float, onlstm_chunk_sz: int = 10,
                     bidirectional: bool = False,
                     ) -> list[UnifiedRNN]:
        from ..modules.torch_rnn_wrapper import TorchRNNWrapper as RNNWrapper
        from ..modules.variational_dropout import VariationalDropout

        def _get_cell_in(floor):
            if floor == 0:
                return inp_sz
            if bidirectional:
                return hid_sz * 2
            return hid_sz

        def _get_h_vd(d): return VariationalDropout(d, on_the_fly=False) if d > 0 else None

        rnns = cls = None
        if cell_type == 'onlstm':
            from ..onlstm.onlstm import ONLSTMCell
            rnns = [ONLSTMCell(_get_cell_in(floor), hid_sz, onlstm_chunk_sz, _get_h_vd(h_dropout))
                    for floor in range(num_layers)]

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
    """Static method for different kinds of encoders, and an instance method"""
    @staticmethod
    def get_stacked_rnn_encoder(encoder_type: str, inp_sz, hid_sz, num_layers, dropout=0, num_heads=12) -> StackEncoder:
        from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper as PTRNNWrapper
        from allennlp.modules.seq2seq_encoders import AugmentedLstmSeq2SeqEncoder
        from allennlp.modules.seq2seq_encoders import StackedBidirectionalLstmSeq2SeqEncoder

        def _get_inp_sz(floor: int, bidirectional=False):
            if floor == 0:
                return inp_sz
            elif bidirectional:
                return hid_sz * 2
            else:
                return hid_sz

        enc_classes = dict(
            lstm=lambda floor: PTRNNWrapper(nn.LSTM(_get_inp_sz(floor), hid_sz, batch_first=True)),
            gru=lambda floor: PTRNNWrapper(nn.GRU(_get_inp_sz(floor), hid_sz, batch_first=True)),
            bilstm=lambda floor: PTRNNWrapper(nn.LSTM(_get_inp_sz(floor, True), hid_sz, batch_first=True,
                                                      bidirectional=True)),
            bigru=lambda floor: PTRNNWrapper(nn.GRU(_get_inp_sz(floor, True), hid_sz, batch_first=True,
                                                    bidirectional=True)),
            torch_bilstm=lambda floor: ExtLSTM(nn.LSTM(_get_inp_sz(floor, True), hid_sz, batch_first=True,
                                                       bidirectional=True)),
            aug_lstm=lambda floor: AugmentedLstmSeq2SeqEncoder(
                _get_inp_sz(floor), hid_sz,
                recurrent_dropout_probability=dropout, use_highway=True,
            ),
            aug_bilstm=lambda floor: StackedBidirectionalLstmSeq2SeqEncoder(
                _get_inp_sz(floor, True), hid_sz, num_layers=1,
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
        encoder = EncoderStacker([enc_cls(floor) for floor in range(num_layers)],
                                 input_size=inp_sz,
                                 output_size=hid_sz,
                                 input_dropout=dropout)
        return encoder

    @staticmethod
    def get_diora_encoder(encoder_type: str, inp_sz, hid_sz, concat_outside: bool = False) -> StackEncoder:
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
    def get_perturb_parse_gcn_encoder(inp_sz, hid_sz, num_layers) -> StackEncoder:
        from ..perturb_and_parse.pp_encoder import PerturbParseEncoder
        enc = PerturbParseEncoder(inp_sz, hid_sz, num_layers)
        return enc

    @classmethod
    def get_stacked_cell_encoder(cls, enc_type, inp_sz, hid_sz, num_layers, dropout,
                                 use_bid=False, use_pseq=False) -> StackEncoder:
        rnns = cls.get_rnn_list(enc_type, inp_sz, hid_sz, num_layers, dropout)
        b_rnns = cls.get_rnn_list(enc_type, inp_sz, hid_sz, num_layers, dropout)
        from .cell_encoder import CellEncoder
        return EncoderStacker([CellEncoder(rnn, brnn, use_pseq)
                               if use_bid else
                               CellEncoder(rnn, None, use_pseq)
                               for rnn, brnn in zip(rnns, b_rnns)],
                              input_dropout=dropout)

    def get_encoder_stack(self) -> StackEncoder | None:
        p = self.p
        dropout = getattr(p, 'enc_dropout', getattr(p, 'dropout', 0.))
        hid_sz = getattr(p, 'enc_out_dim', p.hidden_sz)
        num_heads = getattr(p, 'num_heads', 16 if hid_sz % 16 == 0 else 10)

        if 'syn_gcn' == p.encoder:  # syn_gcn doesn't require a stack encoder but only a bundle
            return None
        elif p.encoder.startswith('plm:'):  # encoder is a pretrained huggingface model
            return None

        if 'diora' in p.encoder:
            concat = getattr(p, 'diora_concat_outside', False)
            return self.get_diora_encoder(p.encoder, p.emb_sz, hid_sz, concat)

        if 'perturb_parse' == p.encoder:
            return self.get_perturb_parse_gcn_encoder(p.emb_sz, hid_sz, p.num_enc_layers)

        if getattr(p, 'use_cell_based_encoder', False):
            return self.get_stacked_cell_encoder(p.encoder, p.emb_sz, hid_sz, p.num_enc_layers, dropout,
                                                 getattr(p, 'cell_encoder_is_bidirectional', False),
                                                 getattr(p, 'cell_encoder_uses_packed_sequence', False))

        return self.get_stacked_rnn_encoder(p.encoder, p.emb_sz, hid_sz, p.num_enc_layers, dropout, num_heads)


class EmbEncBundleMixin:
    def get_embed_encoder_bundle(self, emb, enc: StackEncoder | None, padding_idx=0) -> EmbedAndEncode:
        p, vocab = self.p, self.vocab
        hid_sz = getattr(p, 'enc_out_dim', p.hidden_sz)
        dropout = getattr(p, 'enc_dropout', getattr(p, 'dropout', 0.))

        # bundles that do not require an encoder (thus enc=None is fine)
        if p.encoder == 'syn_gcn':
            from .seq_graph_emb_enc import GraphEmbedEncoder
            gcn_act = getattr(p, 'syn_gcn_activation', 'mish')
            emb_enc = GraphEmbedEncoder(emb, p.emb_sz, hid_sz, p.num_enc_layers, gcn_act, enc_dropout=dropout)
            return emb_enc

        elif p.encoder.startswith('plm:'):
            return self.get_pretrained_model_bundle()

        enc_dropout = getattr(p, 'enc_dropout', getattr(p, 'dropout', 0.))
        if 'diora' in p.encoder:
            from ..diora.diora_bundle import SeqEmbedAndDiora
            return SeqEmbedAndDiora(emb, enc, padding_idx, enc_dropout,
                                    use_diora_loss=getattr(p, 'diora_loss_enabled', False))

        from .seq_embed_encode import SeqEmbedEncoder
        emb_enc = SeqEmbedEncoder(emb, enc, padding_idx, enc_dropout)
        emb_enc = self.get_seq_compound_bundle(emb_enc)
        return emb_enc

    def get_pretrained_model_bundle(self):
        logging.getLogger().info('Using pretrained encoder model, thus the source embedding is'
                                 'not used and never shared with the target embedding.')
        from .seq_plm_emb_enc import SeqPLMEmbedEncoder
        p = self.p
        model_name: str = p.encoder[4:]     # excluding the prefix 'plm:'
        emb_enc = SeqPLMEmbedEncoder(model_name)
        return emb_enc

    def get_seq_compound_bundle(self, emb_enc: EmbedAndEncode):
        p, vocab = self.p, self.vocab
        # a compound EmbEnc is an EmbEnc that depends on another EmbEnc.
        compound_emb_enc = getattr(p, 'compound_encoder', None)

        if compound_emb_enc == 'cpcfg':
            from ..pcfg.pcfg_emb_enc import PCFGEmbedEncode
            from ..pcfg.C_PCFG import CompoundPCFG
            return PCFGEmbedEncode(
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
            from ..pcfg.pcfg_emb_enc import PCFGEmbedEncode
            from ..pcfg.C_PCFG import ReducedCPCFG
            return PCFGEmbedEncode(
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
            from ..pcfg.pcfg_emb_enc import PCFGEmbedEncode
            from ..pcfg.TN_PCFG import TNPCFG
            return PCFGEmbedEncode(
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
            from ..pcfg.pcfg_emb_enc import PCFGEmbedEncode
            from ..pcfg.TN_PCFG import ReducedTDPCFG
            return PCFGEmbedEncode(
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


class Seq2SeqBuilder(EmbeddingMixin,
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

    def get_model(self, cls=None) -> BaseSeq2Seq:
        from trialbot.data import START_SYMBOL, END_SYMBOL

        p, vocab = self.p, self.vocab
        emb_sz = p.emb_sz

        source_embedding, target_embedding = self.get_embeddings()
        encoder = self.get_encoder_stack()
        embed_and_encoder = self.get_embed_encoder_bundle(source_embedding, encoder, padding_idx=0)
        if cls is not None:
            cls = cls
        elif isinstance(embed_and_encoder, EmbedAndGraphEncode):
            cls = SynGraph2Seq
        else:
            cls = BaseSeq2Seq

        enc_out_dim = embed_and_encoder.get_output_dim()
        dec_in_dim = getattr(p, 'dec_in_dim', p.hidden_sz)
        dec_out_dim = getattr(p, 'dec_out_dim', p.hidden_sz)
        proj_in_dim = getattr(p, 'proj_in_dim', p.hidden_sz)

        trans_module, _usages = self._get_transformation_module(embed_and_encoder.is_bidirectional(), enc_out_dim, dec_out_dim)
        trans_usage_string = self._get_trans_usage_string(trans_module, _usages)

        # Initialize attentions. And compute the dimension requirements for all attention modules
        # the encoder attention size depends on the transformation usage
        enc_attn_sz = dec_out_dim if trans_usage_string in ('consistent', 'attn') else enc_out_dim
        enc_attn = get_attention(p.enc_attn, dec_out_dim, enc_attn_sz)
        # dec output attend over previous dec outputs, thus attn_context dimension == dec_output_dim
        dec_hist_attn = get_attention(p.dec_hist_attn, dec_out_dim, dec_out_dim)
        attn_sz = 0 if dec_hist_attn is None else dec_out_dim
        attn_sz += 0 if enc_attn is None else enc_attn_sz
        dec_inp_composer = get_attn_composer(p.dec_inp_composer, attn_sz, emb_sz, dec_in_dim, p.dec_inp_comp_activation)
        proj_inp_composer = get_attn_composer(p.proj_inp_composer, attn_sz, dec_out_dim, proj_in_dim, p.proj_inp_comp_activation)

        if enc_attn is not None or dec_hist_attn is not None:
            assert dec_inp_composer is not None and proj_inp_composer is not None, "Attention must be composed"

        dec_dropout = getattr(p, 'dec_dropout', getattr(p, 'dropout', 0.))

        rnns = self.get_rnn_list(p.decoder, dec_in_dim, dec_out_dim, p.num_dec_layers, dec_dropout,
                                 getattr(p, 'dec_onlstm_chunk_sz', 10))
        decoder = RNNCellStacker(rnns, dec_dropout)

        word_proj = self.get_word_projection(proj_in_dim, target_embedding)

        model = cls(
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
            attn_supervision=getattr(p, 'attn_supervision', 'none')
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

    @classmethod
    def base_hparams(cls):
        from trialbot.training.hparamset import HyperParamSet
        from trialbot.utils.root_finder import find_root
        p = HyperParamSet.common_settings(find_root())
        p.TRAINING_LIMIT = 150
        p.WEIGHT_DECAY = 0.
        p.OPTIM = "adabelief"
        p.ADAM_BETAS = (0.9, 0.999)
        p.batch_sz = 16

        p.lr_scheduler_kwargs = {'model_size': 400, 'warmup_steps': 50}
        p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"
        p.src_namespace = 'source_tokens'
        p.tgt_namespace = 'target_tokens'

        p.hidden_sz = 300
        p.dropout = .5
        p.decoder = "lstm"
        p.max_decoding_step = 100
        p.scheduled_sampling = .1

        p.num_enc_layers = 1
        p.num_dec_layers = 1

        p.tied_decoder_embedding = False
        p.emb_sz = 100

        p.enc_out_dim = p.hidden_sz
        p.dec_in_dim = p.hidden_sz
        p.dec_out_dim = p.hidden_sz

        p.enc_dec_trans_usage = 'consistent'
        p.enc_dec_trans_act = 'mish'
        p.enc_dec_trans_forced = True

        p.proj_in_dim = p.emb_sz

        p.enc_dropout = 0
        p.dec_dropout = 0.5
        p.enc_attn = "dot_product"
        p.dec_hist_attn = "none"
        p.dec_inp_composer = 'cat_mapping'
        p.dec_inp_comp_activation = 'mish'
        p.proj_inp_composer = 'cat_mapping'
        p.proj_inp_comp_activation = 'mish'

        p.decoder_init_strategy = "forward_last_all"
        p.encoder = 'bilstm'
        p.use_cell_based_encoder = False
        # cell-based encoders: typed_rnn, ind_rnn, onlstm, lstm, gru, rnn; see models.base_s2s.base_seq2seq.py file
        # seq-based encoders: lstm, transformer, bilstm, aug_lstm, aug_bilstm; see models.base_s2s.stacked_encoder.py file
        p.cell_encoder_is_bidirectional = True     # any cell-based RNN encoder above could be bidirectional
        p.cell_encoder_uses_packed_sequence = False

        return p

