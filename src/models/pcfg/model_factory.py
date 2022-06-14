
from models.base_s2s.model_factory import EncoderStackMixin, EmbEncBundleMixin, EmbeddingMxin
from models.pcfg.C_PCFG import CompoundPCFG
from models.pcfg.TN_PCFG import TNPCFG
from models.pcfg.seq2pcfg import Seq2PCFG


class Seq2PCFGBuilder(EmbeddingMxin,
                      EncoderStackMixin,
                      EmbEncBundleMixin,
                      ):
    def __init__(self, p, vocab):
        super().__init__()
        self.p = p
        self.vocab = vocab

    def get_model(self):
        emb = self.get_source_embedding()
        enc = self.get_encoder_stack()
        emb_enc = self.get_embed_encoder_bundle(emb, enc, padding_idx=0)

        model = Seq2PCFG(emb_enc, self.get_pcfg(z_dim=emb_enc.get_output_dim()))
        return model

    def get_pcfg(self, z_dim: int):
        p, vocab = self.p, self.vocab

        decoder = getattr(p, 'decoder', None)
        if decoder == 'cpcfg':
            pcfg = CompoundPCFG(
                num_nonterminal=p.num_pcfg_nt,
                num_preterminal=p.num_pcfg_pt,
                num_vocab_token=vocab.get_vocab_size(p.tgt_namespace),
                hidden_sz=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                z_dim=z_dim,
                encoder_input_dim=None,
            )

        elif decoder == 'tdpcfg':
            pcfg = TNPCFG(
                rank=getattr(p, 'td_pcfg_rank', p.num_pcfg_nt // 10),
                num_nonterminal=p.num_pcfg_nt,
                num_preterminal=p.num_pcfg_pt,
                num_vocab_token=vocab.get_vocab_size(p.tgt_namespace),
                hidden_sz=getattr(p, 'pcfg_hidden_dim', p.hidden_sz),
                z_dim=z_dim,
                encoder_input_dim=None,
            )

        else:
            raise NotImplementedError

        return pcfg

    @classmethod
    def from_param_and_vocab(cls, p, vocab):
        return cls(p, vocab).get_model()
