from typing import Optional
import torch
from torch import nn
from torch.nn import init
from allennlp.nn.util import masked_softmax
from ..interfaces.attention import Attention

class GeneralizedDotProductAttention(Attention):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, attend_over, attend_mask = None) -> torch.Tensor:
        """
        :param inputs:      (...a..., ...b..., vec_dim)
        :param attend_over: (...a..., num_tensors, attn_dim)
        :param attend_mask: (...a..., num_tensors)
        :return: context vector: (...a..., ...b..., attn_dim)
        """
        input_size = inputs.size()
        bunch_size = attend_over.size()
        assert attend_over.ndim >= 3
        attn_prefix_dims = bunch_size[:-2]
        attn_prefix_len = attend_over.ndim - 2
        assert inputs.ndim >= attn_prefix_len + 1   # dims of ...b... could be absent
        assert inputs.size()[:attn_prefix_len] == attn_prefix_dims
        input_suffix_dims = input_size[attn_prefix_len:-1]

        # rs_inputs: (...a..., -1, vec_dim, 1)
        rs_inputs = inputs.reshape(*attn_prefix_dims, -1, input_size[-1], 1)

        # rs_a: (...a..., 1, num_tensors, attn_dim)
        rs_a = attend_over.unsqueeze(-3)

        # similarity: (...a..., -1, num_tensors, 1)
        similarity = torch.matmul(rs_a, rs_inputs)

        # rs_mask: (...a..., 1, num_tensors, 1)
        rs_mask = None
        if attend_mask is not None:
            rs_mask = attend_mask.unsqueeze(-2).unsqueeze(-1)

        # attn_weights: (...a..., -1, num_tensors, 1)
        attn_weights = masked_softmax(similarity, rs_mask, dim=-2)

        # rs_context: (...a..., -1, attn_dim)
        rs_context = (attn_weights * rs_a).sum(-2)
        # context: (...a..., ...b..., attn_dim)
        context = rs_context.reshape(*attn_prefix_dims, *input_suffix_dims, -1)

        return context
