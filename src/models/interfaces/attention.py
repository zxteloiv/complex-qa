from abc import ABC
import torch
from .composer import TwoVecComposer


class _AttnWeightMixin:
    def get_latest_attn_weights(self) -> torch.Tensor:
        if self._last_attn_weights is None:
            raise ValueError('Attention module has never been applied.')
        return self._last_attn_weights

    def save_last_attn_weights(self, weights):
        self._last_attn_weights = weights

    def __init__(self):
        super().__init__()
        self._last_attn_weights = None


class VectorContextComposer(TwoVecComposer):
    """
    How to combine the context vector and the hidden states then?
    """
    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class AdaptiveAttention(torch.nn.Module, _AttnWeightMixin, ABC):
    def forward(self,
                inputs: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: torch.LongTensor | None = None,
                graph_mask: torch.LongTensor | None = None,
                ) -> torch.Tensor:
        """
        Although the typing is the same as the superclass due to the limited,
        an adaptive attention is able to support either a token or matrix for the input.
        thus the tensor shape may vary, so is the output context or attn of the attention.

        :param inputs: (batch, input_length, input_dim) or simply (batch, input_dim)
        :param attend_over: (batch, attend_length, attend_dim)
        :param attend_mask: (batch, attend_length)
        :param graph_mask: (batch, input_length, attend_length)
        :return: context: (batch, input_length, attend_dim) or simply (batch, input_dim)
        """
        raise NotImplementedError


class AdaptiveAttnLogits(torch.nn.Module):
    """
    Returns the unormalized scores for attention.
    Used in some customized attention modules.
    The input tensor can be either a token or a matrix.
    When the input is a matrix, the interface is the same as MatrixAttention in AllenNLP.
    """
    def forward(self, inputs: torch.Tensor, attend_over: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (batch, input_dim) or (batch, M, input_dim)
        :param attend_over: (batch, attend_len, attend_dim)
        :return: attn: (batch, attend_len) or (batch, M, attend_len)
        """
        is_vec_attn = inputs.ndim < 3
        if is_vec_attn:
            inputs = inputs.unsqueeze(1)   # (batch, 1, input_dim)

        attn = self.matrix_attn_logits(inputs, attend_over)

        if is_vec_attn:
            attn = attn.squeeze(1)     # (batch, attend_len)

        return attn

    def matrix_attn_logits(self, inputs, attend_over) -> torch.Tensor:
        """
        Implement the attention for matrix, the dimension should be reshaped by the caller.
        :param inputs: (batch, M, input_dim)
        :param attend_over: (batch, N, attend_dim)
        :return: attn: (batch, M, N)
        """
        raise NotImplementedError

