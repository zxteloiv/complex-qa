from typing import Optional, Tuple
from allennlp.modules.attention import Attention
from ..transformer.multi_head_attention import GeneralMultiHeadAttention
from ..interfaces.attention import Attention as IAttn
import torch


class AllenNLPAttentionWrapper(IAttn):
    """
    A wrapper for matrix attention in allennlp, fitting the interface of the multi-headed attention
    defined in models.transformer.multi_head_attention
    """
    def __init__(self, attn: Attention, attn_dropout: float = 0.):
        super(AllenNLPAttentionWrapper, self).__init__()
        self._attn: Attention = attn
        self._dropout = torch.nn.Dropout(attn_dropout)

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

        # attn: (batch, max_attend_length, -1)
        attn = self._attn(inputs, attend_over, attend_mask)
        attn = self._dropout(attn).unsqueeze(-1)

        # context: (batch, attend_dim)
        context = (attn * attend_over).sum(1)

        return context


class SingleTokenMHAttentionWrapper(IAttn):
    def __init__(self, attn: GeneralMultiHeadAttention):
        super(SingleTokenMHAttentionWrapper, self).__init__()
        self._attn = attn

    def forward(self, inputs, attend_over, attend_mask = None):
        """
        Do a multi-head attention for _input_ tokens over the _attend_over_ tokens.
        _attend_mask_ is used to wipe out padded tokens in the corresponding sequences.

        :param inputs: (batch, input_dim)
        :param attend_over: (batch, max_attend_length, attend_dim)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return: Tuple of context vector and attention vector:
                   context: (batch, output_dim)
                 attention: (batch, num_heads, max_attend_length)
        """
        # inputs: (batch, 1, input_dim)
        inputs = inputs.unsqueeze(1)

        c, a = self._attn(inputs, attend_over, attend_mask)

        c = c.squeeze(1)
        a = a.squeeze(1)

        return c

