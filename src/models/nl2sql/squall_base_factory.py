from torch import nn
from models.base_s2s.model_factory import EncoderStackMixin
from models.base_s2s.stacked_rnn_cell import StackedRNNCell
from models.modules.variational_dropout import VariationalDropout as VDrop
from models.modules.res import ResLayer
from models.modules.attention_wrapper import get_wrapped_attention as get_attn


class SquallBaseBuilder(EncoderStackMixin):
    def __init__(self, p, vocab):
        super().__init__()
        self.p = p
        from trialbot.data import NSVocabulary
        self.vocab: NSVocabulary = vocab

    def get_model(self):
        from .squall_base import SquallBaseParser
        from transformers import AutoModel
        p, vocab = self.p, self.vocab
        hid_sz = p.hidden_sz
        dropout = p.dropout

        plm_model = AutoModel.from_pretrained(p.plm_model)
        plm2hidden = nn.Sequential(
            nn.Linear(plm_model.config.hidden_size, hid_sz),
            ResLayer(hid_sz, hid_sz),
        )
        h_ctx_enc = self.get_stacked_rnn_encoder(p.plm_encoder, hid_sz * 2, p.plm_enc_out, 2, dropout=0)

        tgt_type_keys: tuple = ('pad', 'keyword', 'column', 'literal_string', 'literal_number')
        parser = SquallBaseParser(
            plm_model=plm_model,
            hidden_sz=p.hidden_sz,
            plm2hidden=plm2hidden,
            hidden_enc=h_ctx_enc,
            kwd_embedding=nn.Sequential(
                nn.Embedding(vocab.get_vocab_size(p.ns_keyword), p.emb_sz),
                VDrop(dropout, on_the_fly=False),
                nn.Linear(p.emb_sz, hid_sz),
                VDrop(dropout, on_the_fly=False),
            ),
            col2input=nn.Sequential(ResLayer(hid_sz, hid_sz), VDrop(dropout, on_the_fly=False)),
            span2input=nn.Sequential(ResLayer(hid_sz * 2, hid_sz), VDrop(dropout, on_the_fly=False)),
            decoder=StackedRNNCell(
                self.get_stacked_rnns(p.decoder, hid_sz, hid_sz, p.num_dec_layers, dropout),
                dropout=dropout,    # vertical dropout between cell layers
            ),
            sql_word_attn=get_attn(p.word_ctx_attn, hid_sz, hid_sz),
            sql_col_attn=get_attn(p.col_ctx_attn, hid_sz, hid_sz),
            sql_type=nn.Sequential(
                nn.Linear(hid_sz * 3, hid_sz),
                nn.Mish(),
                VDrop(dropout, on_the_fly=True),
                nn.Linear(hid_sz, len(tgt_type_keys))
            ),
            sql_keyword=nn.Sequential(
                nn.Linear(hid_sz * 3, hid_sz),
                nn.Mish(),
                VDrop(dropout, on_the_fly=True),
                nn.Linear(hid_sz, vocab.get_vocab_size(p.ns_keyword))
            ),
            sql_col_type=nn.Sequential(
                nn.Linear(hid_sz * 3, hid_sz),
                nn.Mish(),
                VDrop(dropout, on_the_fly=True),
                nn.Linear(hid_sz, vocab.get_vocab_size(p.ns_coltype))
            ),
            sql_col_copy=get_attn(p.col_copy, hid_sz * 3, hid_sz, num_heads=p.num_heads, mha_mean_weight=True),
            sql_span_begin=get_attn(p.span_begin, hid_sz * 3, hid_sz, num_heads=p.num_heads, mha_mean_weight=True),
            sql_span_end=get_attn(p.span_end, hid_sz * 3, hid_sz, num_heads=p.num_heads, mha_mean_weight=True),
            tgt_type_keys=tgt_type_keys,
            decoder_init_strategy=p.decoder_init,
        )
        return parser

    @classmethod
    def from_param_vocab(cls, p, vocab):
        """
        example conf:

        p.emb_sz = 256
        p.hidden_sz = 200
        p.plm_model = 'bert-base-uncased'

        p.ns_keyword = 'keyword'
        p.ns_col_type = 'col_type'

        p.dropout = 0.2

        p.decoder = 'lstm'
        p.num_dec_layers = 2

        p.word_ctx_attn = 'dot_product'
        p.col_ctx_attn = 'dot_product'

        p.num_heads = 10  # for Multi-Head Attention only
        p.col_copy = 'mha'
        p.span_begin = 'bilinear'
        p.span_end = 'bilinear'

        p.decoder_init = 'zero_all'

        :param p: hyper param set
        :param vocab: NSVocabulary
        :return:
        """
        return cls(p, vocab).get_model()


