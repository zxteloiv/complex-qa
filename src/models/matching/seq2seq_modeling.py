import torch
from torch import nn
from allennlp.nn.util import masked_softmax
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

class Seq2SeqModeling(nn.Module):
    def __init__(self,
                 a_embedding,
                 b_embedding,
                 encoder: PytorchSeq2SeqWrapper,
                 decoder: PytorchSeq2SeqWrapper,
                 a_padding,
                 b_padding,
                 prediction,
                 attention,
                 ):
        super().__init__()
        self.a_embedding = a_embedding
        self.b_embedding = b_embedding
        self.encoder: PytorchSeq2SeqWrapper = encoder
        self.decoder: PytorchSeq2SeqWrapper = decoder
        self.padding_val_a = a_padding
        self.padding_val_b = b_padding
        self.prediction = prediction
        self.attention = attention

    def forward(self, sent_a: torch.Tensor, sent_b: torch.Tensor):
        input_b = sent_b[:, :-1].contiguous()
        target_b = sent_b[:, 1:].contiguous()

        a = self.a_embedding(sent_a)
        b = self.b_embedding(input_b)
        mask_a = (sent_a != self.padding_val_a).long()
        mask_b = (input_b != self.padding_val_b).long()

        logits = self.forward_emb(a, b, mask_a, mask_b)
        loss = nn.functional.cross_entropy(logits.transpose(1, 2), target_b)
        return loss, logits

    def forward_emb(self, a, b, mask_a, mask_b):
        # a_hid: (batch, a_len, a_hid)
        a_hid = self.encoder(a, mask_a)
        try:
            final_states = self._encoder._states
        except AttributeError:
            import sys
            raise Exception("The wrapper must be stateful to get final states")

        # b_hid: (batch, b_len, b_hid)
        b_hid = self.decoder(b, mask_b, final_states)

        # attn: (batch, a_len, b_len)
        # weight_a: (batch, a_len, b_len)
        attn = self.attention(a_hid, b_hid)
        weight_a = masked_softmax(attn, mask_a.unsqueeze(2), dim=1)
        # ctx_for_b: (batch, b_len, a_hid) <- (batch, b_len, a_len) x (batch, a_len, a_hid)
        ctx_for_b = weight_a.transpose(1, 2).matmul(a_hid)

        pred_input = torch.cat([b_hid, ctx_for_b], dim=-1)
        # logits: (batch, b_len, num_classes)
        logits = self.prediction(pred_input)
        return logits
