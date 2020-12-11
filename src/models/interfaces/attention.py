from typing import Optional, Tuple
import torch


class Attention(torch.nn.Module):
    def forward(self,
                inputs: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrap the Attention in AllenNLP, with sufficient dimension and context value computation

        :param inputs: (batch, input_dim)
        :param attend_over: (batch, max_attend_length, attend_dim)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return: Tuple of context vector and attention vector:
                   context: (batch, max_input_length, output_dim=attend_dim)
                 attention: (batch, max_input_length, 1, max_attend_length)
        """
        raise NotImplementedError
