import math
from typing import Optional, Union, Tuple
import torch
from torch import nn
from torch.nn import init
from allennlp.nn.util import masked_softmax
from ..interfaces.attention import Attention


class GeneralizedDotProductAttention(Attention):
    def forward(self, inputs, attend_over, attend_mask = None, structural_mask = None) -> torch.Tensor:
        """
        :param inputs:      (...a..., ...b..., vec_dim)
        :param attend_over: (...a..., num_tensors, attn_dim)
        :param attend_mask: (...a..., num_tensors)
        :param structural_mask: (...a..., ...b..., num_tensors)
        :return: context vector: (...a..., ...b..., attn_dim)
        """
        input_size, bunch_size = inputs.size(), attend_over.size()
        assert attend_over.ndim >= 3
        attn_prefix_dims, attn_prefix_len = bunch_size[:-2], attend_over.ndim - 2
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

        if structural_mask is not None:
            # s_mask: (...a..., -1, num_tensors, 1)
            rs_s_mask = structural_mask.reshape(*attn_prefix_dims, -1, bunch_size[-2], 1)
            # rs_mask: (...a..., -1, num_tensors, 1)
            rs_mask = rs_mask * rs_s_mask

        # attn_weights: (...a..., -1, num_tensors, 1)
        attn_weights = masked_softmax(similarity, rs_mask, dim=-2)

        # rs_context: (...a..., -1, attn_dim)
        rs_context = (attn_weights * rs_a).sum(-2)
        # context: (...a..., ...b..., attn_dim)
        context = rs_context.reshape(*attn_prefix_dims, *input_suffix_dims, -1)

        self.save_last_attn_weights(attn_weights.reshape(*attn_prefix_dims, *input_suffix_dims, -1))
        return context


class GeneralizedBilinearAttention(Attention):
    def __init__(self, attn_dim: int, vec_dim: int,
                 use_linear: bool = True,
                 use_bias: bool = True,
                 activation: Optional[nn.Module] = None,
                 eval_top1_ctx: bool = False,
                 ):
        super().__init__()

        self.bi_weight = nn.Parameter(torch.zeros(attn_dim, vec_dim))

        if use_linear:
            self.a_linear = nn.Linear(attn_dim, 1, bias=False)
            self.b_linear = nn.Linear(vec_dim, 1, bias=False)
        else:
            self.a_linear = self.b_linear = None

        # without a nonlinear activation, the bias is useless because the softmax will erase the differences
        self.bias = nn.Parameter(torch.zeros(1,)) if use_bias and activation is not None else None
        self.activation = activation
        self.attn_dim = attn_dim
        self.vec_dim = vec_dim
        self.eval_top1_ctx = eval_top1_ctx

        self.reset_parameters()

    def extra_repr(self) -> str:
        return 'input_size={}, attn_size={}, linear={}, bias={}'.format(
            self.bi_weight.size(1), self.bi_weight.size(0), self.a_linear is not None, self.bias is not None,
        )

    def reset_parameters(self) -> None:
        # nonlinearity = self._check_nonlinearity(self.activation)
        # init.kaiming_uniform_(self.bi_weight, a=math.sqrt(5), nonlinearity=nonlinearity)
        init.kaiming_uniform_(self.bi_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.bi_weight)
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
            else:
                bound = 0
            # gain = init.calculate_gain(nonlinearity, math.sqrt(5))
            # std = gain / math.sqrt(fan_in)
            # bound = std * math.sqrt(3.0) / math.sqrt(fan_in)
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

    def forward(self, inputs, attend_over, attend_mask = None) -> torch.Tensor:
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
        rs_inputs = inputs.view(*attn_prefix_dims, -1, input_size[-1])

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
        rs_mask = None
        if attend_mask is not None:
            rs_mask = attend_mask.unsqueeze(-2).unsqueeze(-1)

        # the mask and the similarity are ensured to have the same shape, thus getting rid of broadcasting
        # attn: (...a..., -1, num_tensors, 1)
        attn_weights = masked_softmax(similarity, rs_mask, dim=-2)

        if self.eval_top1_ctx and not self.training:
            # rs_a: (...a..., 1, num_tensors, attn_dim)
            # max_pos: (...a..., -1(=prod(...b...)), 1, 1)
            max_pos = attn_weights.argmax(dim=-2, keepdim=True)

            attn_size = max_pos.size()
            ctx_size = rs_a.size()

            # pos_idx: (...a..., prod(input_suffix_dims), 1, attn_dim)
            pos_idx = max_pos.expand(*attn_prefix_dims, -1, -1, ctx_size[-1])
            ctx_input = rs_a.expand(*attn_prefix_dims, attn_size[-3], -1, -1)
            context = torch.gather(ctx_input, dim=-2, index=pos_idx)
            context = context.view(*attn_prefix_dims, *input_suffix_dims, -1)

        else:
            # rs_context: (...a..., -1, attn_dim)
            # context: (...a..., ...b..., attn_dim)
            rs_context = (attn_weights * rs_a).sum(-2)
            context = rs_context.view(*attn_prefix_dims, *input_suffix_dims, -1)

        attn_weights = attn_weights.squeeze(-1).reshape(*attn_prefix_dims, *input_suffix_dims, -1)
        self.save_last_attn_weights(attn_weights)

        return context

