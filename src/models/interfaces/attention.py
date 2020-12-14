from typing import Optional, Tuple
import torch


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


class AttentionWeight(torch.nn.Module):
    """Compute the attention similarities only,
    NOTE the returned value is not context vector but the attention weights"""
    def forward(self,
                inputs: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Wrap the Attention in AllenNLP, with sufficient dimension and context value computation

        :param inputs: (batch, input_dim)
        :param attend_over: (batch, max_attend_length, attend_dim)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return: attention: (batch, max_input_length, max_attend_length)
        """
        raise NotImplementedError
