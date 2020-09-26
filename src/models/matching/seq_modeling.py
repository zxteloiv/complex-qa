import torch
from torch import nn
from allennlp.nn.util import masked_softmax
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from utils.nn import seq_cross_ent, seq_likelihood, filter_cat
from allennlp.training.metrics.perplexity import Perplexity

class SeqModeling(nn.Module):
    def __init__(self,
                 embedding,
                 encoder: PytorchSeq2SeqWrapper,
                 padding,
                 prediction,
                 attention,
                 ):
        super().__init__()
        self.embedding = embedding
        self.encoder: PytorchSeq2SeqWrapper = encoder
        self.padding_val = padding
        self.prediction = prediction
        self.attention = attention
        self.ppl = Perplexity()

    def forward(self, seq: torch.Tensor):
        inp = seq[:, :-1].contiguous()
        tgt = seq[:, 1:].contiguous()

        seq = self.embedding(inp)
        mask = (inp != self.padding_val).long()

        logits = self.forward_emb(seq, mask)
        xent = seq_cross_ent(logits, tgt, mask)
        self.ppl(xent)
        output = {"likelihoods": seq_likelihood(logits, tgt, mask)}
        if self.training:
            output['loss'] = xent
        return output

    def get_metric(self, reset=False):
        ppl = self.ppl.get_metric(reset)
        return {"PPL": ppl}

    def forward_emb(self, sent, mask):
        # a_hid: (batch, len, hid)
        hid = self.encoder(sent, mask)

        ctx = None
        if self.attention is not None:
            # attn: (batch, len, len)
            # weight_a: (batch, len, len)
            attn = self.attention(hid, hid)
            weight = masked_softmax(attn, mask.unsqueeze(2), dim=1)
            # ctx_for_b: (batch, len, hid) <- (batch, len, len) x (batch, len, hid)
            ctx = weight.transpose(1, 2).matmul(hid)

        pred_input = filter_cat([hid, ctx], dim=-1)
        # logits: (batch, len, num_classes)
        logits = self.prediction(pred_input)
        return logits

