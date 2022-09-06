from torch import nn

from models.base_s2s.model_factory import EncoderStackMixin
from models.base_s2s.stacked_rnn_cell import StackedRNNCell
from models.modules.variational_dropout import VariationalDropout as VDrop, VariationalDropout
from models.modules.attention_wrapper import AdaptiveGeneralAttention, AdaptiveAllenLogits
from allennlp.modules.matrix_attention import MatrixAttention


class SquallBaseBuilder(EncoderStackMixin):
    def __init__(self, p, vocab):
        super().__init__()
        self.p = p
        from trialbot.data import NSVocabulary, START_SYMBOL, END_SYMBOL
        self.vocab: NSVocabulary = vocab
        self.start_tok = START_SYMBOL
        self.end_tok = END_SYMBOL

    def get_mha(self, vector_dim, matrix_dim, is_sparse: bool = False):
        p = self.p
        num_heads = self.p.num_heads
        attn = AdaptiveGeneralAttention(
            AdaptiveAllenLogits(MatrixAttention.by_name('dot_product')()),
            init_tau=(vector_dim // num_heads) ** 0.5,
            min_tau=p.min_tau,
            num_heads=num_heads,
            pre_q_mapping=nn.Linear(vector_dim, matrix_dim),
            pre_k_mapping=nn.Linear(matrix_dim, matrix_dim),
            pre_v_mapping=nn.Linear(matrix_dim, matrix_dim),
            post_ctx_mapping=nn.Linear(matrix_dim, matrix_dim),
            training_ctx=p.attn_training_ctx,
            eval_ctx=p.attn_eval_ctx,
            tau_is_fixed=p.attn_weight_policy != 'tau_schedule',
            is_sparse=is_sparse,
        )
        return attn

    def get_bilinear(self, vector_dim, matrix_dim, use_mapping: bool = True, is_sparse: bool = False):
        p = self.p
        if use_mapping:
            hid = min(vector_dim, matrix_dim)
            attn = AdaptiveGeneralAttention(
                AdaptiveAllenLogits(MatrixAttention.by_name('bilinear')(hid, hid)),
                pre_q_mapping=nn.Sequential(nn.Linear(vector_dim, hid), nn.Mish()),
                pre_k_mapping=nn.Sequential(nn.Linear(matrix_dim, hid), nn.Mish()),
                pre_v_mapping='shared',
                init_tau=p.init_tau,
                min_tau=p.min_tau,
                training_ctx=p.attn_training_ctx,
                eval_ctx=p.attn_eval_ctx,
                tau_is_fixed=p.attn_weight_policy != 'tau_schedule',
                is_sparse=is_sparse,
            )
        else:
            attn = AdaptiveGeneralAttention(
                AdaptiveAllenLogits(MatrixAttention.by_name('bilinear')(vector_dim, matrix_dim)),
                init_tau=p.init_tau,
                min_tau=p.min_tau,
                training_ctx=p.attn_training_ctx,
                eval_ctx=p.attn_eval_ctx,
                tau_is_fixed=p.attn_weight_policy != 'tau_schedule',
                is_sparse=is_sparse,
            )
        return attn

    def get_model(self):
        from .squall_base import SquallBaseParser
        from transformers import AutoModel
        p, vocab = self.p, self.vocab
        hid_sz = p.hidden_sz
        dropout = p.dropout

        word_enc = self.get_stacked_rnn_encoder(p.plm_encoder, hid_sz * 2, p.plm_enc_out, p.plm_enc_layers,
                                                dropout=.2)
        col_enc = self.get_stacked_rnn_encoder(p.plm_encoder, hid_sz * 2, p.plm_enc_out, p.plm_enc_layers,
                                               dropout=.2)
        assert word_enc.get_output_dim() == hid_sz and col_enc.get_output_dim() == hid_sz

        plm_model = AutoModel.from_pretrained(p.plm_model)
        tgt_type_keys: tuple = ('pad', 'keyword', 'column', 'literal_string', 'literal_number')
        parser = SquallBaseParser(
            plm_model=plm_model,
            word_plm2hidden=nn.Sequential(
                nn.Linear(plm_model.config.hidden_size, hid_sz, bias=False),
            ),
            col_plm2hidden=nn.Sequential(
                nn.Linear(plm_model.config.hidden_size, hid_sz, bias=False),
            ),
            word_enc=word_enc,
            col_enc=col_enc,
            word_col_attn=self.get_bilinear(hid_sz, hid_sz, use_mapping=False, is_sparse=p.is_sparse),
            col_word_attn=self.get_bilinear(hid_sz, hid_sz, use_mapping=False, is_sparse=p.is_sparse),
            kwd_embedding=nn.Sequential(
                nn.Embedding(vocab.get_vocab_size(p.ns_keyword), p.emb_sz),
                VDrop(dropout, on_the_fly=False),
            ) if p.emb_sz == hid_sz else nn.Sequential(
                nn.Embedding(vocab.get_vocab_size(p.ns_keyword), p.emb_sz),
                VDrop(dropout, on_the_fly=False),
                nn.Linear(p.emb_sz, hid_sz),
                VDrop(dropout, on_the_fly=False),
            ),
            col2input=nn.Sequential(
                nn.Linear(hid_sz, hid_sz // 2),
                VDrop(dropout, on_the_fly=False),
                nn.Mish(),
                nn.Linear(hid_sz // 2, hid_sz),
            ),
            span2input=nn.Sequential(
                nn.Linear(hid_sz * 2, hid_sz // 2),
                VDrop(dropout, on_the_fly=False),
                nn.Mish(),
                nn.Linear(hid_sz // 2, hid_sz),
            ),
            decoder=StackedRNNCell(
                self.get_stacked_rnns(p.decoder, hid_sz, hid_sz, p.num_dec_layers, dropout),
                dropout=dropout,    # vertical dropout between cell layers
            ),
            sql_word_attn=self.get_mha(hid_sz, hid_sz, is_sparse=p.is_sparse),
            sql_col_attn=self.get_bilinear(hid_sz, hid_sz, use_mapping=False, is_sparse=p.is_sparse),
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
            sql_col_copy=self.get_bilinear(hid_sz * 3, hid_sz),
            sql_span_begin=self.get_bilinear(hid_sz * 3, hid_sz),
            sql_span_end=self.get_bilinear(hid_sz * 3, hid_sz),
            start_token=self.start_tok,
            end_token=self.end_tok,
            ns_keyword=p.ns_keyword,
            ns_coltype=p.ns_coltype,
            vocab=vocab,
            p_tuning=self.get_p_tuning_module(plm_model.config),
            tgt_type_keys=tgt_type_keys,
            decoder_init_strategy=p.decoder_init,
            attn_weight_policy=p.attn_weight_policy,
        )
        return parser

    def get_p_tuning_module(self, plm_config):
        from .p_tuning_v2 import PTuningV2Prompt
        from transformers import BertConfig
        plm_config: BertConfig
        p = self.p
        prefix_len = getattr(p, 'prompt_length', 0)
        if not isinstance(prefix_len, int) or prefix_len <= 0:
            return None

        prompt_encoder = getattr(p, 'prompt_encoder', False)
        if prompt_encoder:
            enc_out_size = plm_config.num_hidden_layers * 2 * plm_config.hidden_size
            mlp_hid = enc_out_size // plm_config.num_attention_heads
            prompt_encoder = nn.Sequential(
                nn.Linear(plm_config.hidden_size,  mlp_hid),
                VariationalDropout(p.dropout),
                nn.SELU(),
                nn.Linear(mlp_hid, enc_out_size),
                nn.SELU()   # no dropout at the final layer
            )

        p_tuning = PTuningV2Prompt(
            prefix_len=p.prompt_length,
            n_layers=plm_config.num_hidden_layers,
            n_head=plm_config.num_attention_heads,
            plm_hidden=plm_config.hidden_size,
            prefix_enc=prompt_encoder if prompt_encoder else None,
            dropout=nn.Dropout(p.dropout),
        )
        return p_tuning

    @classmethod
    def from_param_vocab(cls, p, vocab):
        """
        example conf:

        p.emb_sz = 256
        p.hidden_sz = 256
        p.plm_model = 'bert-base-uncased'

        p.ns_keyword = 'keyword'
        p.ns_col_type = 'col_type'

        p.dropout = 0.2

        p.decoder = 'lstm'
        p.num_dec_layers = 2

        p.plm_encoder = 'aug_bilstm'
        p.plm_enc_out = p.hidden_sz // 2  # = hid_sz or hid_sz//2 when encoder is bidirectional
        p.plm_enc_layers = 1

        p.num_heads = 1     # heads for attention only
        p.decoder_init = 'zero_all'

        :param p: hyper param set
        :param vocab: NSVocabulary
        :return:
        """
        return cls(p, vocab).get_model()


