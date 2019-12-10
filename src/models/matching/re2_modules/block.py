import torch
from torch import nn

class Re2Block(nn.Module):
    def __init__(self,
                 a_encoder,
                 b_encoder,
                 a_fusion,
                 b_fusion,
                 alignment,
                 dropout: float = 0.2,
                 ):
        super().__init__()
        self.a_enc = a_encoder
        self.b_enc = b_encoder
        self.a_fusion = a_fusion
        self.b_fusion = b_fusion
        self.alignment = alignment
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                a: torch.Tensor,
                b: torch.Tensor,
                mask_a: torch.LongTensor,
                mask_b: torch.LongTensor):
        raw_a, raw_b = a, b
        a = self.a_enc(a, mask_a)
        b = self.b_enc(b, mask_b)

        a = torch.cat([raw_a, a], dim=-1)
        b = torch.cat([raw_b, b], dim=-1)

        align_a, align_b = self.alignment(a, b, mask_a, mask_b)
        a = self.a_fusion(a, align_a)
        b = self.b_fusion(b, align_b)
        return a, b

