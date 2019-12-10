from typing import Tuple
import torch
from torch import nn
from .dense import Re2Dense
import math
from allennlp.nn.util import masked_softmax

class Re2Alignment(nn.Module):
    def __init__(self, inp_sz: int, hid_sz: int, mode: str, activation=nn.ReLU()):
        super().__init__()
        tau = nn.Parameter(torch.randn(size=(), ), requires_grad=True)
        nn.init.constant_(tau, val=math.sqrt(1 / hid_sz))
        self.tau = tau

        self.mode = mode
        if mode == "linear":
            self.a_linear = Re2Dense(inp_sz, hid_sz, activation=activation)
            self.b_linear = Re2Dense(inp_sz, hid_sz, activation=activation)
        elif mode == "bilinear":
            from allennlp.modules.matrix_attention import BilinearMatrixAttention
            self.a_linear = Re2Dense(inp_sz, hid_sz, activation=activation)
            self.b_linear = Re2Dense(inp_sz, hid_sz, activation=activation)
            self.bilinear = BilinearMatrixAttention(hid_sz, hid_sz, activation=activation, use_input_biases=True)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                mask_a: torch.LongTensor, mask_b: torch.LongTensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param a: (batch, len_a, inp_sz)
        :param b: (batch, len_b, inp_sz)
        :param mask_a: (batch, len_a)
        :param mask_b: (batch, len_b)
        :return:
        """
        # attn_strength: (batch, len_a, len_b)
        if self.mode == "linear":
            attn_strength = self._linear_attn(a, b)
        elif self.mode == "bilinear":
            attn_strength = self._bilinear_attn(a, b)
        else:
            attn_strength = self._identity_attn(a, b)

        # weights (a and b): (batch, len_a, len_b)
        weight_a = masked_softmax(attn_strength, mask_a.unsqueeze(2), dim=1)
        weight_b = masked_softmax(attn_strength, mask_b.unsqueeze(1), dim=2)

        # align_a: (batch, len_a, inp_sz)
        # align_b: (batch, len_b, inp_sz)
        align_a = weight_b.matmul(b)
        align_b = weight_a.transpose(1, 2).matmul(a)
        return align_a, align_b

    def _identity_attn(self, a: torch.Tensor, b: torch.Tensor):
        return torch.bmm(a, b.transpose(1, 2)) * self.tau

    def _linear_attn(self, a: torch.Tensor, b: torch.Tensor):
        a = self.a_linear(a)
        b = self.b_linear(b)
        return self._identity_attn(a, b)

    def _bilinear_attn(self, a, b):
        a = self.a_linear(a)
        b = self.b_linear(b)
        return self.bilinear(a, b) * self.tau

