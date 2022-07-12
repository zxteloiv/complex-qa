from typing import Optional, Tuple
from allennlp.modules.attention import Attention as AllenAttention
from allennlp.nn.util import masked_softmax
from ..transformer.multi_head_attention import GeneralMultiHeadAttention
from ..interfaces.attention import Attention as IAttn
import torch


class AllenNLPAttentionWrapper(IAttn):
    """
    A wrapper for matrix attention in allennlp, fitting the interface of the multi-headed attention
    defined in models.transformer.multi_head_attention
    """
    def get_latest_attn_weights(self) -> torch.Tensor:
        if self._attn_weights is None:
            raise ValueError('Attention weight is None.')
        return self._attn_weights

    def __init__(self, attn: AllenAttention, attn_dropout: float = 0.,
                 use_temperature_schedule: bool = False,
                 init_tau: float = 1., min_tau: float = .05):
        super(AllenNLPAttentionWrapper, self).__init__()
        self._attn: AllenAttention = attn
        self._dropout = torch.nn.Dropout(attn_dropout)
        self.tau = init_tau
        self.init_tau = init_tau
        self.min_tau = min_tau
        self.use_schedule = use_temperature_schedule
        if use_temperature_schedule:
            # manually normalize in the forward pass of the wrapper
            self._attn._normalize = False
        self._attn_weights = None

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

        # attn, weights: (batch, max_attend_length)
        attn = self._attn(inputs, attend_over, attend_mask)
        if self.use_schedule:
            if self.training:
                temperature = max(self.tau, self.min_tau)
            else:
                temperature = self.min_tau
            weights = masked_softmax(attn / temperature, attend_mask.bool(), dim=-1)
        else:
            weights = attn  # no schedule, the attention has been processed of mask softmax

        self._attn_weights = weights    # (batch, max_attend_length)

        # context: (batch, attend_dim)
        context = (self._dropout(weights.unsqueeze(-1)) * attend_over).sum(1)
        return context


class SingleTokenMHAttentionWrapper(IAttn):
    def get_latest_attn_weights(self) -> torch.Tensor:
        if self._attn_weights is None:
            raise ValueError("Attention Weight is None")
        return self._attn_weights

    def __init__(self, attn: GeneralMultiHeadAttention, mean_reduction_heads: bool = False):
        super(SingleTokenMHAttentionWrapper, self).__init__()
        self._attn = attn
        self._attn_weights = None
        self._mean_on_heads = mean_reduction_heads

    def forward(self, inputs, attend_over, attend_mask = None, structural_mask = None):
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

        # c: (batch, 1, output_dim)
        # a: (batch, 1, num_heads, max_attend_length)
        c, a = self._attn(inputs, attend_over, attend_mask, structural_mask)

        c = c.squeeze(1)
        a = a.squeeze(1)

        if self._mean_on_heads:
            self._attn_weights = a.mean(1)  # (b, max_attend_length)
        else:
            self._attn_weights = a  # (b, num_heads, max_attend_length)
        return c


class PreAttnMappingWrapper(IAttn):
    def get_latest_attn_weights(self) -> torch.Tensor:
        return self.attn.get_latest_attn_weights()

    def __init__(self, attn: IAttn, input_map: torch.nn.Module = None, attend_map: torch.nn.Module = None):
        super().__init__()
        self.attn = attn
        self.input_map = input_map
        self.attend_map = attend_map

    def forward(self,
                inputs: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: Optional[torch.LongTensor] = None) -> torch.Tensor:
        inputs = self.input_map(inputs) if self.input_map else inputs
        attend_over = self.attend_map(attend_over) if self.attend_map else attend_over
        return self.attn(inputs, attend_over, attend_mask)


def get_wrapped_attention(attn_type: str,
                          vector_dim: int = 0,
                          matrix_dim: int = 0,
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

    elif attn_type == 'cosine':
        from allennlp.modules.attention import CosineAttention
        attn = CosineAttention()
        attn = AllenNLPAttentionWrapper(attn)

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
        attn = SingleTokenMHAttentionWrapper(attn, mean_reduction_heads=kwargs.get('mha_mean_weight', False))

    elif attn_type == "seq_mha":
        from ..transformer.multi_head_attention import GeneralMultiHeadAttention
        num_heads = kwargs.get('num_heads', 8)
        attn = GeneralMultiHeadAttention(num_heads,
                                         input_dim=vector_dim,
                                         total_attention_dim=vector_dim,
                                         total_value_dim=vector_dim,
                                         attend_to_dim=matrix_dim,
                                         output_dim=matrix_dim,
                                         attention_dropout=attention_dropout,)

    elif attn_type == "none":
        attn = None

    else:
        raise NotImplementedError

    return attn

