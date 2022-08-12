import logging
from typing import Optional, Tuple, Union

from allennlp.modules.attention import Attention as AllenAttention
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.nn.util import masked_softmax
from ..transformer.multi_head_attention import GeneralMultiHeadAttention
from ..interfaces.attention import Attention as IAttn, AdaptiveAttention, AdaptiveAttnLogits
import torch


class AdaptiveAllenLogits(AdaptiveAttnLogits):
    """
    By wrapping the allennlp matrix, we can avoid to inherit the AdaptiveAttnLogits class
    and implement dot-prod and bilinear attentions on our own.
    And the MatrixAttention is not adaptive, the wrapper can help to achieve this.
    """
    def __init__(self, allen_mat_attn: MatrixAttention):
        super().__init__()
        self.mat_attn = allen_mat_attn

    def matrix_attn_logits(self, inputs, attend_over) -> torch.Tensor:
        return self.mat_attn(inputs, attend_over)


class AdaptiveGeneralAttention(AdaptiveAttention):
    """Most Sophisticated attention features are implemented here,
    independent with the attention score computations.
    A generalization of dot-prod, bilinear, and multihead attentions.
    """
    def __init__(self,
                 attn_scorer: AdaptiveAttnLogits,
                 init_tau: float = 1,
                 min_tau: float = 1e-12,
                 num_heads: int = 1,
                 pre_q_mapping: torch.nn.Module = None,
                 pre_k_mapping: torch.nn.Module = None,
                 pre_v_mapping: torch.nn.Module = None,
                 post_ctx_mapping: torch.nn.Module = None,
                 training_ctx: str = 'weighted_sum',
                 eval_ctx: str = 'weighted_sum',
                 shared_scorer_for_heads: bool = True,
                 ):
        """

        :param attn_scorer: Compute the unnormalized attention score.
        :param init_tau: set the temperature for softmax.
        :param min_tau: set the minimal temperature for softmax.
        :param num_heads: int, 1 by default. if greater than 1, multi-head attention is applied.
        :param pre_q_mapping: map the input to Query by some mapping e.g. (linear)
        :param pre_k_mapping: map the attn targets to Keys by some mapping.
        :param pre_v_mapping: map the attn targets to Values by some mapping.
        :param post_ctx_mapping: map the output context after the attentions.
        :param training_ctx: weighted_sum, argmax, Hungarian, sample, or normed-topK.
        :param eval_ctx: weighted_sum, argmax, Hungarian, sample, or normed-topK.
        :param shared_scorer_for_heads: True by default. otherwise the attn_scorer needs
         to be copied into a different module (but with the same attention type) for each head.
         Since the multihead is usually meant to use with dot-prod attention, we fix it as True
         and leave the function in the future.
        """
        super().__init__()
        self.attn_scorer = attn_scorer
        self.min_tau = min_tau
        self.tau = init_tau
        self.init_tau = init_tau
        self.n_head = num_heads

        def _identity(x): return x

        self.pre_q_mapping = pre_q_mapping or _identity
        self.pre_k_mapping = pre_k_mapping or _identity
        self.pre_v_mapping = pre_v_mapping or _identity
        self.post_ctx_mapping = post_ctx_mapping or _identity

        self.training_ctx = training_ctx
        self.eval_ctx = eval_ctx

        self.shared_head_scorer = shared_scorer_for_heads
        if not self.shared_head_scorer:
            raise NotImplementedError('Since the multihead is usually meant to use '
                                      'with dot-prod attention, we fix it as True '
                                      'and leave the function in the future.')

    def extra_repr(self) -> str:
        return 'heads={}, init_tau={}, min_tau={}, training_ctx={}, eval_ctx={}'.format(
            self.n_head, self.init_tau, self.min_tau, self.training_ctx, self.eval_ctx
        )

    def update_tau(self, tau):
        logging.getLogger(self.__class__.__name__).info(
            f'Update the temperature from {self.tau} to {tau}'
        )
        self.tau = tau

    def forward(self,
                inputs: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        :param inputs: (batch, M, input_dim) or simply (batch, input_dim)
        :param attend_over: (batch, N, attend_dim)
        :param attend_mask: (batch, N)
        """
        bsz = inputs.size(0)
        is_vec_attn = inputs.ndim < 3

        # (bsz * head, [qkv]_length, d_[qkv])
        query, key, value = self._pre_attn_process(inputs, attend_over)
        value = value.view(bsz, self.n_head, *value.size()[-2:])

        attn = self.attn_scorer(query, key)    # (b * head, M, N)
        attn_logit = attn.view(bsz, self.n_head, *attn.size()[-2:]) # (b, head, M, N)
        attn_prob = self._compute_weight(attn_logit, attend_mask)   # (b, head, M, N)

        # ctx: (b, M, d)
        ctx = self._weighted_sum_ctx(attn_prob, value)
        ctx = self.post_ctx_mapping(ctx)

        if is_vec_attn:
            attn_prob = attn_prob.squeeze(-2)   # remove the M=1 dimension if required
            ctx = ctx.squeeze(-2)
        attn_prob = attn_prob.mean(dim=1)   # remove the multihead dimension;

        self.save_last_attn_weights(attn_prob)
        return ctx

    def _pre_attn_process(self, inputs, attend_over):
        assert inputs.ndim in (2, 3)

        # pre-attn mappings
        query = self.pre_q_mapping(inputs)          # (b, [M,], d_q)
        key = self.pre_k_mapping(attend_over)       # (b, N, d_k)
        value = self.pre_v_mapping(attend_over)     # (b, N, d_v)

        def _split_heads(t: torch.Tensor):
            if t.ndim == 3:
                bsz, t_len, hid = t.size()
            else:
                bsz, hid = t.size()
                t_len = 1

            assert hid % self.n_head == 0
            head_t = t.reshape(bsz, t_len, self.n_head, hid // self.n_head)
            head_t = head_t.transpose(1, 2).reshape(bsz * self.n_head, t_len, hid // self.n_head)
            return head_t

        # processing multiheads
        headed_query, headed_key, headed_val = map(_split_heads, (query, key, value))
        return headed_query, headed_key, headed_val

    def _compute_weight(self, attn_logit, attn_mask=None, graph_mask=None, head_graph_mask=None):
        """
        Compute the attention probabilities from logits, given various masks.

        :param attn_logit: (batch, head, M, N)
        :param attn_mask: (batch, N), used to wipe out paddings in the attn targets.
        :param graph_mask: (batch, M, N), to inject graph connections of attention inputs ant targets.
        :param head_graph_mask: (batch, head, M, N), to inject connections for individual heads.
        :return: attn_prob: (batch, head, M, N), the last dim sums to 1
        """
        bsz, nhead, inp_len, attn_len = attn_logit.size()
        mask = torch.ones_like(attn_logit, dtype=torch.long)
        if attn_mask is not None:
            mask *= attn_mask.view(bsz, 1, 1, attn_len)
        if graph_mask is not None:
            mask *= graph_mask.view(bsz, 1, inp_len, attn_len)
        if head_graph_mask is not None:
            mask *= head_graph_mask

        attn_prob = masked_softmax(attn_logit, mask.bool())
        return attn_prob

    def _weighted_sum_ctx(self, attn_prob, value):
        """
        :param attn_prob: (batch, head, M, N)
        :param value: (batch, head, N, d_v)
        :return:
        """
        ctx = torch.einsum('bhmn,bhnd->bmhd', attn_prob, value) # (batch, M, head, d_v)
        ctx = ctx.view(*ctx.size()[:2], -1)     # (batch, M, d)
        return ctx


class AllenNLPAttentionWrapper(IAttn):
    """
    A wrapper for matrix attention in allennlp, fitting the interface of the multi-headed attention
    defined in models.transformer.multi_head_attention
    """
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

        self.save_last_attn_weights(weights)    # (batch, max_attend_length)

        # context: (batch, attend_dim)
        context = (self._dropout(weights.unsqueeze(-1)) * attend_over).sum(1)
        return context


class SingleTokenMHAttentionWrapper(IAttn):
    def __init__(self, attn: GeneralMultiHeadAttention, mean_reduction_heads: bool = False):
        super(SingleTokenMHAttentionWrapper, self).__init__()
        self._attn = attn
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
            self.save_last_attn_weights(a.mean(1))  # (b, max_attend_length)
        else:
            self.save_last_attn_weights(a)  # (b, num_heads, max_attend_length)
        return c


class PreAttnMappingWrapper(IAttn):
    def get_latest_attn_weights(self) -> torch.Tensor:
        return self.attn.get_latest_attn_weights()

    def __init__(self, attn: IAttn, input_map: torch.nn.Module = None,
                 attend_map: torch.nn.Module = None):
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
        from .adaptive_attention import GeneralizedBilinearAttention
        from torch import nn
        use_linear = kwargs.get('use_linear', True)
        use_bias = kwargs.get('use_bias', True)
        use_argmax = kwargs.get('use_argmax', False)
        activation = nn.Tanh() if kwargs.get('use_tanh_activation', False) else None
        attn = GeneralizedBilinearAttention(matrix_dim, vector_dim,
                                            activation=activation,
                                            use_linear=use_linear,
                                            use_bias=use_bias,
                                            eval_top1_ctx=use_argmax,
                                            )

    elif attn_type == "generalized_dot_product":
        from .adaptive_attention import GeneralizedDotProductAttention
        attn = GeneralizedDotProductAttention()

    elif attn_type == "dot_product":
        from allennlp.modules.attention import DotProductAttention
        attn = DotProductAttention()
        attn = AllenNLPAttentionWrapper(attn, attention_dropout)

    elif attn_type == "mha":
        from ..transformer.multi_head_attention import GeneralMultiHeadAttention
        num_heads = kwargs.get('num_heads', 8)
        use_argmax = kwargs.get('use_argmax', False)
        mean_reduction_heads = kwargs.get('mha_mean_weight', False)
        attn = GeneralMultiHeadAttention(num_heads,
                                         input_dim=vector_dim,
                                         total_attention_dim=vector_dim,
                                         total_value_dim=vector_dim,
                                         attend_to_dim=matrix_dim,
                                         output_dim=matrix_dim,
                                         attention_dropout=attention_dropout,
                                         eval_top1_ctx=use_argmax,
                                         )
        attn = SingleTokenMHAttentionWrapper(attn, mean_reduction_heads=mean_reduction_heads)

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

    elif attn_type == 'adaptive_dot_product':
        from allennlp.modules.matrix_attention import DotProductMatrixAttention
        attn = AdaptiveGeneralAttention(AdaptiveAllenLogits(DotProductMatrixAttention()))

    elif attn_type == 'adaptive_bilinear':
        from allennlp.modules.matrix_attention import BilinearMatrixAttention
        attn = AdaptiveGeneralAttention(
            AdaptiveAllenLogits(BilinearMatrixAttention(vector_dim, matrix_dim)),
        )

    elif attn_type == 'adaptive_mha':
        from allennlp.modules.matrix_attention import DotProductMatrixAttention
        num_heads = kwargs.get('num_heads', 8)
        attn = AdaptiveGeneralAttention(
            AdaptiveAllenLogits(DotProductMatrixAttention()),
            init_tau=(vector_dim // num_heads) ** 0.5,
            num_heads=num_heads,
            pre_q_mapping=torch.nn.Linear(vector_dim, matrix_dim),
            pre_k_mapping=torch.nn.Linear(matrix_dim, matrix_dim),
            pre_v_mapping=torch.nn.Linear(matrix_dim, matrix_dim),
            post_ctx_mapping=torch.nn.Linear(matrix_dim, matrix_dim),
        )

    elif attn_type == "none":
        attn = None

    else:
        raise NotImplementedError

    return attn

