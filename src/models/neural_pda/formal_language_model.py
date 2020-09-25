import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nn import seq_cross_ent, seq_likelihood
from .npda import NeuralPDA
from allennlp.training.metrics.perplexity import Perplexity

class NPDAFLM(nn.Module):
    def __init__(self, npda: NeuralPDA, ntdec_factor: float = 1.):
        super().__init__()
        self.npda = npda
        self.beta = ntdec_factor
        self.perplexity = Perplexity()

    def forward(self, seq: torch.LongTensor):
        """
        :param seq: (batch, length)
        :return:
        """
        seq_in = seq[:, :-1].contiguous()
        seq_out = seq[:, 1:].contiguous()

        # logits: (batch, step, vocab)
        # pushes: (batch, step, 2)
        # raw_codes: (batch, step, 2, hidden_dim)
        # valid_logits: (batch, step, 3)
        tlogits, pushes, raw_codes, vlogits = self.npda(seq_in)

        tgt_mask = (seq_out != self.npda.pad_id).long()

        output = {}
        cross_ent = seq_cross_ent(tlogits, seq_out, tgt_mask)
        if self.training:
            quantized_codes = self.npda.codebook[pushes].detach()
            loss_ntdec = (quantized_codes - raw_codes).norm()
            loss = cross_ent + self.beta * loss_ntdec
            output['loss'] = loss

        self.perplexity(cross_ent)
        output['likelihoods'] = seq_likelihood(tlogits, seq_out, tgt_mask)
        return output

    def get_metric(self, reset=False):
        ppl = self.perplexity.get_metric(reset)
        return {"PPL": ppl}

