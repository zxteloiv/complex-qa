from typing import Optional, Tuple
import torch
from .composer import TwoVecComposer


class Attention(torch.nn.Module):
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

    def get_latest_attn_weights(self) -> torch.Tensor:
        raise NotImplementedError


class VectorContextComposer(TwoVecComposer):
    """
    How to combine the context vector and the hidden states then?
    """
    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError
