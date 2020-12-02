import torch
from torch import nn
from allennlp.nn.util import masked_softmax
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from utils.nn import seq_cross_ent, seq_likelihood, filter_cat
from allennlp.training.metrics.perplexity import Perplexity, Average

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
        self.err = Average()

    def forward(self, seq: torch.Tensor):
        inp = seq[:, :-1].contiguous()
        tgt = seq[:, 1:].contiguous()

        seq = self.embedding(inp)
        mask = (tgt != self.padding_val).long()

        logits = self.forward_emb(seq, mask)
        batch_xent = seq_cross_ent(logits, tgt, mask, average=None)
        for instance_xent in batch_xent:
            self.ppl(instance_xent)
        xent = batch_xent.mean()
        preds = logits.argmax(dim=-1)
        total_err = ((preds != tgt) * mask).sum(list(range(mask.ndim))[1:]) > 0
        for instance_err in total_err:
            self.err(instance_err)

        output = {"likelihoods": seq_likelihood(logits, tgt, mask), "preds": preds}
        if self.training:
            output['loss'] = xent
        return output

    def get_metric(self, reset=False):
        ppl = self.ppl.get_metric(reset)
        err = self.err.get_metric(reset)
        return {"PPL": ppl, "ERR": err}

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

