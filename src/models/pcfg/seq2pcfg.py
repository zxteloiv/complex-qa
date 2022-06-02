import logging

import torch
from allennlp.training.metrics import Average
from torch import nn

from models.interfaces.encoder import EmbedAndEncode
from models.interfaces.loss_module import LossModule
from models.interfaces.metrics_module import MetricsModule
from models.pcfg.C_PCFG import CompoundPCFG
from utils.nn import aggregate_layered_state


class Seq2PCFG(nn.Module):
    def __init__(self,
                 embed_encode: EmbedAndEncode,
                 pcfg: CompoundPCFG,
                 padding: int = 0,
                 enc_policy: str = 'forward_last_all',
                 max_decoding_step: int = 90,
                 ):
        super().__init__()
        self.pcfg = pcfg
        self.emb_enc = embed_encode
        self.enc_policy = enc_policy

        self.src_len = Average()
        self.tgt_len = Average()
        self.err_rate = Average()
        self.item_count = 0
        self.padding = padding
        self.max_decoding_step = max_decoding_step

    def get_metrics(self, reset: bool = False):
        metrics = {"COUNT": self.item_count,
                  "SLEN": self.src_len.get_metric(reset),
                  "TLEN": self.tgt_len.get_metric(reset),
                  "ERR": self.err_rate.get_metric(reset),
                  }
        if reset:
            self.item_count = 0

        if isinstance(self.emb_enc, MetricsModule):
            metrics.update(self.emb_enc.get_metrics(reset))

        return metrics

    def compute_metrics(self, source_tokens, target_tokens, pred, pred_mask):
        def _count(toks, metric_obj):
            mask = self.padding != toks
            for l in mask.sum(1):
                metric_obj(l)

        _count(source_tokens, self.src_len)
        _count(target_tokens, self.tgt_len)
        self.item_count += source_tokens.size()[0]
        self._compute_err(target_tokens, pred, pred_mask)

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
            loss = -logPz.mean()
            if isinstance(self.emb_enc, LossModule):
                loss = loss + self.emb_enc.get_loss()
            output.update(target=target_tokens, loss=loss)

        steps = self.max_decoding_step if target_tokens is None else (target_tokens.size()[1] * 2)
        pred, pred_mask = self.pcfg.generate(pcfg_params, max_steps=steps)
        output.update(pred=pred)
        self.compute_metrics(source_tokens, target_tokens, pred, pred_mask)
        return output

    def _compute_err(self, target: torch.Tensor, pred, pred_mask):
        if target is None:
            return

        target_masks = (target != self.padding).long()

        for tgt, tgt_mask, out, out_mask in zip(target, target_masks, pred, pred_mask):
            if tgt_mask.sum() != out_mask.sum():
                self.err_rate(1)
                continue

            min_len = min(tgt.size()[0], out.size()[0])
            tgt_crop: torch.Tensor = tgt[:min_len]
            out_crop: torch.Tensor = out[:min_len]
            mask_crop: torch.Tensor = tgt_mask[:min_len]
            if (tgt_crop * mask_crop != out_crop * mask_crop).any():
                self.err_rate(1)
                continue

            self.err_rate(0)
