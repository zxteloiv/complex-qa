from typing import Optional, Tuple
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


class Attention(torch.nn.Module, _AttnWeightMixin):
    """Compute the attention and merged for the context value, returns only the context vector"""
    def forward(self,
                inputs: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Wrap the Attention in AllenNLP, with sufficient dimension and context value computation

        :param inputs: (batch, input_dim)
        :param attend_over: (batch, max_attend_length, attend_dim)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return: context: (batch, max_input_length, output_dim=attend_dim)
        """
        raise NotImplementedError


class VectorContextComposer(TwoVecComposer):
    """
    How to combine the context vector and the hidden states then?
    """
    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError
