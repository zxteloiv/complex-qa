from ..transformer.multi_head_attention import GeneralMultiHeadAttention
import torch
from torch import nn

class MHAEncoder(nn.Module):
    def __init__(self, inp_sz, out_sz, num_heads, dropout=.2):
        super().__init__()
        attn = GeneralMultiHeadAttention(num_heads,
                                         input_dim=inp_sz,
                                         total_attention_dim=inp_sz,
                                         total_value_dim=inp_sz,
                                         output_dim=out_sz,
                                         attend_to_dim=inp_sz,
                                         attention_dropout=0.,
                                         )
        norm = nn.LayerNorm(out_sz)
        self.self_attn = attn
        self.norm = norm
        self.input_dim = inp_sz
        self.output_dim = out_sz
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.LongTensor):
        """
        :param x: (batch, N, input_emb)
        :param mask: (batch, N)
        :return: (batch, N, output_emb)
        """
        attn, _ = self.self_attn(input=x, attend_over=x, attend_mask=mask)
        attn = self.norm(attn)
        return self.dropout(attn)

    def is_bidirectional(self):
        return False

