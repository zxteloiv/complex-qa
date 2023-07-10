import torch
from models.interfaces.attention import AdaptiveAttnLogits
import math
from torch.nn import init


class DotProductLogits(AdaptiveAttnLogits):
    def __init__(self, input_dim: int, attend_dim: int):
        super().__init__()
        if input_dim != attend_dim:
            raise ValueError(f'Unequal dimensions {input_dim} and {attend_dim} '
                             f'cannot be applied with dot-product attentions.')

    def matrix_attn_logits(self, inputs: torch.Tensor, attend_over: torch.Tensor) -> torch.Tensor:
        """
        Implement the attention for matrix, the dimension should be reshaped by the caller.
        :param inputs: (batch, M, input_dim)
        :param attend_over: (batch, N, attend_dim)
        :return: attn: (batch, M, N)
        """
        return inputs.matmul(attend_over.transpose(-1, -2))


class BilinearLogits(AdaptiveAttnLogits):
    def __init__(self, input_dim: int, attend_dim: int,
                 use_linear: bool = False,
                 use_bias: bool = True):
        super().__init__()
        raise NotImplementedError
