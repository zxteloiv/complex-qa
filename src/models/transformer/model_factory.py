from torch import nn
from trialbot.data import NSVocabulary, START_SYMBOL, END_SYMBOL, PADDING_TOKEN

from models.base_s2s.model_factory import WordProjMixin, EncoderStackMixin, EmbEncBundleMixin, EmbeddingMxin
from models.transformer.decoder import TransformerDecoder, UniversalTransformerDecoder
from models.transformer.parallel_seq2seq import ParallelSeq2Seq


class TransformerBuilder(EmbeddingMxin,
                         EncoderStackMixin,
                         EmbEncBundleMixin,
                         WordProjMixin,
                         ):
    def __init__(self, p, vocab: NSVocabulary):
        super().__init__()
        self.p = p
        self.vocab = vocab

    def get_model(self):
        p, vocab = self.p, self.vocab
        source_embedding, target_embedding = self.get_embeddings()
        encoder = self.get_encoder_stack()  # could be None
        # since embed_encoder is usually implemented by an embedding and a stacked encoder container,
        # the container will be in charge of keeping dimensions fit.
        embed_and_encoder = self.get_embed_encoder_bundle(source_embedding, encoder, padding_idx=0)

        if p.decoder == 'transformer':
            dec_cls = TransformerDecoder
        elif p.decoder == 'universal_transformer':
            dec_cls = UniversalTransformerDecoder
        else:
            raise NotImplementedError

        decoder = dec_cls(input_dim=p.emb_sz,
                          hidden_dim=p.hidden_sz,
                          attend_to_dim=embed_and_encoder.get_output_dim(),
                          num_layers=p.num_dec_layers,
                          num_heads=p.num_heads,
                          feedforward_dropout=p.dropout,
                          residual_dropout=p.dropout,
                          attention_dropout=getattr(p, 'attention_dropout', 0.),
                          feedforward_hidden_activation=p.nonlinear_activation,
                          )

        model = ParallelSeq2Seq(
            vocab=vocab,
            embed_encoder=embed_and_encoder,
            decoder=decoder,
            target_embedding=target_embedding,
            word_projection=self.get_word_projection(decoder.hidden_dim, target_embedding),
            src_namespace=p.src_namespace,
            tgt_namespace=p.tgt_namespace,
            start_id=vocab.get_token_index(START_SYMBOL, namespace=p.tgt_namespace),
            end_id=vocab.get_token_index(END_SYMBOL, namespace=p.tgt_namespace),
            pad_id=vocab.get_token_index(PADDING_TOKEN, namespace=p.tgt_namespace),
            max_decoding_step=p.max_decoding_len,
            beam_size=getattr(p, 'beam_size', 1),
            diversity_factor=getattr(p, 'diversity_factor', 0.),
            accumulation_factor=getattr(p, 'acc_factor', 1.),
            use_bleu=getattr(p, 'use_bleu', False),
            flooding_bias=getattr(p, 'flooding_bias', -1),
        )

        return model
        pass

    @classmethod
    def from_param_and_vocab(cls, p, vocab: NSVocabulary):
        return TransformerBuilder(p, vocab).get_model()

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
        p.src_namespace = 'source_tokens'
        p.tgt_namespace = 'target_tokens'

        p.encoder = 'bilstm'
        p.use_cell_based_encoder = False
        # cell-based encoders: typed_rnn, ind_rnn, onlstm, lstm, gru, rnn; see models.base_s2s.base_seq2seq.py file
        # seq-based encoders: lstm, transformer, bilstm, aug_lstm, aug_bilstm; see models.base_s2s.stacked_encoder.py file
        p.cell_encoder_is_bidirectional = True  # any cell-based RNN encoder above could be bidirectional
        p.cell_encoder_uses_packed_sequence = False

        p.decoder = 'transformer'

        p.emb_sz = 100
        p.tied_decoder_embedding = False    # only True, when hidden == emb_sz, for now
        p.hidden_sz = 300
        p.num_enc_layers = 1
        p.num_dec_layers = 2
        p.num_heads = 10    # must be able to divide hidden_sz

        p.nonlinear_activation = "mish"
        p.dropout = .5
        p.enc_dropout = 0
        p.dec_dropout = p.dropout

        p.lr_scheduler_kwargs = {'model_size': 400, 'warmup_steps': 50}
        p.src_emb_pretrained_file = "~/.glove/glove.6B.100d.txt.gz"

        p.max_decoding_len = 30
        p.beam_size = 1
        p.diversity_factor = 0.
        p.acc_factor = 1.
        return p

