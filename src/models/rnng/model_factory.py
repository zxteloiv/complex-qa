from torch import nn
from trialbot.data import NSVocabulary, START_SYMBOL

from models.base_s2s.model_factory import WordProjMixin, EncoderStackMixin, EmbEncBundleMixin, EmbeddingMxin
from .seq2rnng import Seq2RNNG
from .rnng import RNNG
from ..base_s2s.stacked_rnn_cell import StackedRNNCell
from ..modules.attentions import get_attn_composer, get_attention
from .rnng_utils import get_target_num_embeddings, get_terminal_boundary, token_to_id


class RNNGBuilder(EmbeddingMxin,
                  EncoderStackMixin,
                  EmbEncBundleMixin,
                  WordProjMixin,
                  ):
    def __init__(self, p, vocab: NSVocabulary):
        super().__init__()
        self.p = p
        self.vocab = vocab

    def get_model(self):
        """
        example conf:

        # self.get_embeddings()
        p.emb_sz = 100
        p.src_namespace = "sent"
        p.src_emb_pretrained_file = None
        p.tgt_namespace = "target"

        # self.get_embed_encoder_bundle()
        p.dropout = 0.2
        p.enc_dropout = 0.
        p.encoder = 'lstm'
        p.diora_loss_enabled = True

        # self.get_encoder_stack()
        p.diora_concat_outside = True
        p.hidden_sz = 200
        p.enc_out_dim = 200
        p.num_heads = 10
        p.use_cell_based_encoder = False
        p.cell_encoder_is_bidirectional = False
        p.cell_encoder_uses_packed_sequence = False
        p.num_enc_layers = 1

        # rnng
        p.rnng_namespaces = ('rnng', 'nonterminal', 'terminal')
        p.root_ns = 'grammar_entry'
        p.dec_dropout = p.dropout
        """
        p, vocab = self.p, self.vocab
        hid_sz = p.hidden_sz
        src_emb = self.get_source_embedding()
        tgt_emb = nn.Embedding(get_target_num_embeddings(vocab, p.rnng_namespaces), p.emb_sz)
        embed_and_encoder = self.get_embed_encoder_bundle(src_emb, self.get_encoder_stack(), padding_idx=0)
        attn = get_attention('bilinear', hid_sz * 3, hid_sz)
        proj_attn_comp = get_attn_composer('cat_mapping', hid_sz, hid_sz * 3, hid_sz, 'tanh')
        dec_dropout = getattr(p, 'dec_dropout', getattr(p, 'dropout', 0.))
        assert vocab.get_vocab_size(p.root_ns) == 1, 'root token must be unique'
        grammar_entry_str = list(vocab.get_token_to_index_vocabulary(p.root_ns).keys())[0]
        rnng = RNNG(
            action_encoder=StackedRNNCell(self.get_stacked_rnns(
                'lstm', hid_sz, hid_sz, 1, dec_dropout,
            )),
            buffer_encoder=StackedRNNCell(self.get_stacked_rnns(
                'lstm', hid_sz, hid_sz, 1, dec_dropout,
            )),
            stack_encoder=self.get_stacked_rnn_encoder('lstm', hid_sz, hid_sz, 1, dec_dropout),
            reducer=self.get_stacked_rnn_encoder('lstm', hid_sz, hid_sz, 1, dec_dropout),
            action_embedding=tgt_emb,
            action_projection=nn.Linear(hid_sz, tgt_emb.weight.size()[0]),
            nt_gen_id_boundary=get_terminal_boundary(vocab, p.rnng_namespaces),
            hidden_size=hid_sz,
            start_id=token_to_id(START_SYMBOL, vocab, p.rnng_namespaces),
            root_id=token_to_id(grammar_entry_str, vocab, p.rnng_namespaces),
            emb_drop=dec_dropout,
            loss_reduction="batch",
        )
        model = Seq2RNNG(
            embed_and_encoder, rnng, nn.Linear(embed_and_encoder.get_output_dim(), hid_sz),
            attn=attn, attn_comp=proj_attn_comp, init_strategy='forward_last_all'
        )
        return model

    @classmethod
    def from_param_and_vocab(cls, p, vocab: NSVocabulary):
        return RNNGBuilder(p, vocab).get_model()

