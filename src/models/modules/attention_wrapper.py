from typing import Optional, Tuple, Literal
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

        # attn: (batch, max_attend_length, 1)
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

        # a: (batch, max_input_length, max_attend_length)
        c, a = self._attn(inputs, attend_over, attend_mask)

        c = c.squeeze(1)
        # a = a.squeeze(1)

        return c


def get_wrapped_attention(attn_type: Literal["bilinear", "dot_product", "mha"],
                          vector_dim: int,
                          matrix_dim: int,
                          attention_dropout: float = 0.,
                          **kwargs):
    """
    Build an Attention module with specified parameters.
    TODO: the ugly factory method needs refactored, such that the caller is not required to know the implementation
    :param attn_type: indicates the attention type, e.g. "bilinear", "dot_product" or "none"
    :param vector_dim: the vector to compute attention
    :param matrix_dim: the bunch of vectors to be attended against (batch, num, matrix_dim)
    :param attention_dropout: the dropout to discard some attention weights
    :return: a torch.nn.Module
    """

    attn_type = attn_type.lower()
    if attn_type == "bilinear":
        from allennlp.modules.attention import BilinearAttention
        attn = BilinearAttention(vector_dim=vector_dim, matrix_dim=matrix_dim)
        attn = AllenNLPAttentionWrapper(attn, attention_dropout)

    elif attn_type == "generalized_bilinear":
        from .generalized_attention import GeneralizedBilinearAttention
        from torch import nn
        use_linear = kwargs.get('use_linear', True)
        use_bias = kwargs.get('use_bias', True)
        activation = nn.Tanh() if kwargs.get('use_tanh_activation', False) else None
        attn = GeneralizedBilinearAttention(matrix_dim, vector_dim,
                                            activation=activation, use_linear=use_linear, use_bias=use_bias)

    elif attn_type == "generalized_dot_product":
        from .generalized_attention import GeneralizedDotProductAttention
        attn = GeneralizedDotProductAttention()

    elif attn_type == "dot_product":
        from allennlp.modules.attention import DotProductAttention
        attn = DotProductAttention()
        attn = AllenNLPAttentionWrapper(attn, attention_dropout)

    elif attn_type == "mha":
        from ..transformer.multi_head_attention import GeneralMultiHeadAttention
        num_heads = kwargs.get('num_heads', 8)
        attn = GeneralMultiHeadAttention(num_heads,
                                         input_dim=vector_dim,
                                         total_attention_dim=vector_dim,
                                         total_value_dim=vector_dim,
                                         attend_to_dim=matrix_dim,
                                         output_dim=matrix_dim,
                                         attention_dropout=attention_dropout,)
        attn = SingleTokenMHAttentionWrapper(attn)

    elif attn_type == "none":
        attn = None

    else:
        raise NotImplementedError

    return attn

