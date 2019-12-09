import torch
from torch import nn
from allennlp.nn.util import masked_softmax
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

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

    def forward(self, sent: torch.Tensor):
        inp = sent[:, :-1].contiguous()
        tgt = sent[:, 1:].contiguous()

        sent = self.embedding(inp)
        mask = (inp != self.padding_val_a).long()

        logits = self.forward_emb(sent, mask)
        loss = nn.functional.cross_entropy(logits.transpose(1, 2), tgt)
        return loss

    def forward_emb(self, sent, mask):
        # a_hid: (batch, len, hid)
        hid = self.encoder(sent, mask)
        # attn: (batch, len, len)
        # weight_a: (batch, len, len)
        attn = self.attention(hid, hid)
        weight = masked_softmax(attn, mask.unsqueeze(2), dim=1)
        # ctx_for_b: (batch, len, hid) <- (batch, len, len) x (batch, len, hid)
        ctx = weight.transpose(1, 2).matmul(hid)

        pred_input = torch.cat([hid, ctx], dim=-1)
        # logits: (batch, len, num_classes)
        logits = self.prediction(pred_input)
        return logits

