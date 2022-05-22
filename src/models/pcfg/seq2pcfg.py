import torch
from allennlp.training.metrics import Average
from torch import nn

from models.interfaces.encoder import EmbedAndEncode
from models.interfaces.loss_module import LossModule
from models.pcfg.C_PCFG import CompoundPCFG
from utils.nn import aggregate_layered_state


class Seq2PCFG(nn.Module):
    def __init__(self,
                 embed_encode: EmbedAndEncode,
                 pcfg: CompoundPCFG,
                 padding: int = 0,
                 ):
        super().__init__()
        self.pcfg = pcfg
        self.emb_enc = embed_encode
        self.enc_policy: str = 'forward_last_all'

        self.src_len = Average()
        self.tgt_len = Average()
        self.item_count = 0
        self.padding = padding

    def get_metrics(self, reset: bool = False):
        metric = {"COUNT": self.item_count,
                  "SLEN": self.src_len.get_metric(reset),
                  "TLEN": self.tgt_len.get_metric(reset),
                  }
        if reset:
            self.item_count = 0

        return metric

    def compute_metrics(self, source_tokens, target_tokens):
        def _count(toks, metric_obj):
            mask = self.padding != toks
            for l in mask.sum(1):
                metric_obj(l)

        _count(source_tokens, self.src_len)
        _count(target_tokens, self.tgt_len)
        self.item_count += source_tokens.size()[0]

    def forward(self,
                source_tokens: torch.LongTensor,
                target_tokens: torch.LongTensor = None,
                ):
        layered_state, state_mask = self.emb_enc(source_tokens)
        layered_agg = aggregate_layered_state(layered_state, state_mask)
        pcfg_params = self.pcfg.get_pcfg_params(layered_agg[-1])

        output = {"source": source_tokens}

        if self.training:
            assert target_tokens is not None
            logPz = self.pcfg.inside(target_tokens, pcfg_params)
            loss = -logPz
            if isinstance(self.emb_enc, LossModule):
                loss = loss + self.emb_enc.get_loss()
            output.update(target=target_tokens, loss=loss)

        self.compute_metrics(source_tokens, target_tokens)
        return output
