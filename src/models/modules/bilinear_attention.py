from typing import Optional
import torch
from torch import nn
from torch.nn import init
import math
from allennlp.nn.util import masked_softmax

class GeneralizedBilinearAttention(nn.Module):
    def __init__(self, attn_dim: int, vec_dim: int,
                 use_linear: bool = True,
                 use_bias: bool = True,
                 activation: Optional[nn.Module] = None):
        super().__init__()

        self.bi_weight = nn.Parameter(torch.zeros(attn_dim, vec_dim))

        if use_linear:
            self.a_linear = nn.Linear(attn_dim, 1, bias=False)
            self.b_linear = nn.Linear(vec_dim, 1, bias=False)
        else:
            self.a_linear = self.b_linear = None

        # without a nonlinear activation, the bias is useless because the softmax will erase the differences
        self.bias = nn.Parameter(torch.randn(1,)) if use_bias and activation is not None else None
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nonlinearity = self._check_nonlinearity(self.activation)
        init.kaiming_uniform_(self.bi_weight, a=math.sqrt(5), nonlinearity=nonlinearity)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.bi_weight)
            gain = init.calculate_gain(nonlinearity, math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = std * math.sqrt(3.0) / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _check_nonlinearity(activation: Optional[nn.Module]):
        if activation is None:
            nonlinearity = 'linear'
        elif isinstance(activation, nn.Tanh):
            nonlinearity = 'tanh'
        else:
            nonlinearity = 'leaky_relu'
        return nonlinearity

    def forward(self, inputs, attend_over, attend_mask) -> torch.Tensor:
        """
        Attend over the bunch of vectors.

        The attend_over tensor may have a free-form shape starting with any dimension prefix,
        but ending with the bunch count and attention dimensions.

        The inputs tensor must share the same prefix of the attend_over tensor in its size,
        followed by any dimensions, and ended with the vector dimension.

        The attention will give conducted with the vector dimension and the source bunch attention dimensions.
        The other dimensions are kept untouched.

        Example:
        The module is initialized with a_dim=500, and b_dim=300,
        the parameters share the dimension prefix (128,)
        inputs:      (128,    200, 300)
        attend_over: (128,    25, 500)
        attend_mask: (128,    25)

        the resulted shape will be:
        attention context (128, 200, 500)
        attention weight (128, 200, 25)

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

        # rs_a: (...a..., 1, num_tensors, attn_dim)
        rs_a = attend_over.unsqueeze(-3)

        # rs_aM: (...a..., 1, num_tensors, vec_dim)
        rs_aM = torch.matmul(rs_a, self.bi_weight)

        # rs_inputs: (...a..., -1, vec_dim)
        rs_inputs = inputs.reshape(*attn_prefix_dims, -1, input_size[-1])

        # rs_aMb: (...a..., -1, num_tensors, 1)
        rs_aMb = torch.matmul(rs_aM, rs_inputs.unsqueeze(-1))

        # similarity: (...a..., -1, num_tensors, 1)
        similarity = rs_aMb
        if self.a_linear is not None and self.b_linear is not None:
            # linear weight
            # a: (...a..., -1, num_tensors, 1)
            # b: (...a..., -1,           1, 1)
            linear_weight_a = self.a_linear(rs_a)
            linear_weight_b = self.b_linear(rs_inputs).unsqueeze(-2)
            similarity = similarity + linear_weight_a + linear_weight_b

        if self.bias is not None:
            similarity = similarity + self.bias

        if self.activation:
            similarity = self.activation(similarity)

        # rs_mask: (...a..., 1, num_tensors, 1)
        rs_mask = attend_mask.unsqueeze(-2).unsqueeze(-1)

        # the mask and the similarity are ensured to have the same shape, thus getting rid of broadcasting
        # attn: (...a..., -1, num_tensors, 1)
        attn_weights = masked_softmax(similarity, rs_mask, dim=-2)

        # rs_context: (...a..., -1, attn_dim)
        # context: (...a..., ...b..., attn_dim)
        rs_context = (attn_weights * rs_a).sum(-2)
        context = rs_context.reshape(*attn_prefix_dims, *input_suffix_dims, -1)

        return context






