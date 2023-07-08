import logging
import re
from typing import Callable, Literal

from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.nn.util import masked_softmax

from utils.nn import masked_sparsemax
from ..interfaces.attention import AdaptiveAttention, AdaptiveAttnLogits
import torch


TENSOR_FUNC = Callable[[torch.Tensor], torch.Tensor]


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


def _ctx_by_weighted_sum(attn_prob, value):
    """
    :param attn_prob: (batch, head, M, N)
    :param value: (batch, head, N, d_v)
    :return: (batch, M, d), d == d_v * head
    """
    ctx = torch.einsum('bhmn,bhnd->bmhd', attn_prob, value) # (batch, M, head, d_v)
    ctx = ctx.view(*ctx.size()[:2], -1)     # (batch, M, d)
    return ctx


def _ctx_by_argmax(attn_prob, value):
    """
    :param attn_prob: (batch, head, M, N)
    :param value: (batch, head, N, d_v)
    :return: (batch, M, d), d == d_v * head
    """
    bsz, nhead, inp_len, _ = attn_prob.size()
    max_attn_pos = attn_prob.argmax(dim=-1)  # (b, head, M)

    def tr(*args, **kwargs):
        return torch.arange(*args, **kwargs, device=attn_prob.device)

    max_value = value[tr(bsz).view(-1, 1, 1), tr(nhead).view(1, -1, 1), max_attn_pos]  # (b, head, M, dv)
    ctx = max_value.permute([0, 2, 1, 3]).view(bsz, inp_len, -1)
    return ctx


def _ctx_by_normed_topk(attn_prob, value, num_k: int):
    """
    :param attn_prob: (batch, head, M, N)
    :param value: (batch, head, N, d_v)
    :return: (batch, M, d), d == d_v * head
    """
    def tr(*args, **kwargs):
        return torch.arange(*args, **kwargs, device=attn_prob.device)

    bsz, nhead, inp_len, _ = attn_prob.size()
    k_att, k_idx = torch.topk(attn_prob, num_k, dim=-1)     # (b, head, M, K)
    k_val = value[tr(bsz).view(-1, 1, 1, 1), tr(nhead).view(1, -1, 1, 1), k_idx]  # (b, head, M, K, dv)

    val = (k_att.unsqueeze(-1) * k_val).sum(dim=-2)  # (b, head, M, dv)
    val = val / k_att.sum(dim=-1, keepdim=True)

    ctx = val.permute([0, 2, 1, 3]).reshape(bsz, inp_len, -1)
    return ctx


class AdaptiveGeneralAttention(AdaptiveAttention):
    """Most Sophisticated attention features are implemented here,
    independent with the attention score computations.
    A generalization of dot-prod, bilinear, and multihead attentions.
    """
    def __init__(self,
                 attn_scorer: AdaptiveAttnLogits,
                 init_tau: float = 1,
                 min_tau: float = 1e-8,
                 num_heads: int = 1,
                 pre_q_mapping: torch.nn.Module | None = None,
                 pre_k_mapping: torch.nn.Module | None = None,
                 pre_v_mapping: torch.nn.Module | Literal['shared'] | None = None,
                 post_ctx_mapping: torch.nn.Module = None,
                 training_ctx: str = 'weighted_sum',
                 eval_ctx: str = 'weighted_sum',
                 tau_is_fixed: bool = True,
                 save_head_reduced_attn: bool = True,
                 shared_scorer_for_heads: bool = True,
                 is_sparse: bool = False,
                 ):
        """
        :param attn_scorer: Compute the unnormalized attention score.
        :param init_tau: set the temperature for softmax.
        :param min_tau: set the minimal temperature for softmax.
        :param num_heads: int, 1 by default. if greater than 1, multi-head attention is applied.
        :param pre_q_mapping: map the input to Query by some mapping e.g. (linear)
                default to the identity function if set to None.
        :param pre_k_mapping: map the attn targets to Keys by some mapping.
                default to identity function if set to None.
        :param pre_v_mapping: map the attn targets to Values by some mapping.
                use the same module of pre_k_mapping if set to "shared".
                Otherwise or set to None, it is set to the identity function.
        :param post_ctx_mapping: map the output context after the attentions.
        :param training_ctx: weighted_sum, argmax, or topK_norm, where K is any positive integer.
        :param eval_ctx: weighted_sum, argmax, or topK_norm, where K is any positive integer.
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
        self.tau_is_fixed = tau_is_fixed

        def _identity(x): return x

        self.pre_q_mapping = pre_q_mapping or _identity
        self.pre_k_mapping = pre_k_mapping or _identity

        if isinstance(pre_v_mapping, str) and pre_v_mapping == "shared":
            self.pre_v_mapping = self.pre_k_mapping
        elif isinstance(pre_v_mapping, torch.nn.Module):
            self.pre_v_mapping = pre_v_mapping
        else:
            self.pre_v_mapping = _identity

        self.post_ctx_mapping = post_ctx_mapping or _identity

        self.training_ctx = training_ctx
        self.eval_ctx = eval_ctx

        # a list of hook functions that accept and process the attention weight and returns the modified weights.
        # with hooks we could introduce other dependencies on the weights and even change or rewrite it, for example,
        # using the supervision for the attention weights.
        self._runtime_weight_hooks: list[TENSOR_FUNC] = []
        self._head_graph_mask = None

        self.is_sparse = is_sparse  # use sparsemax or softmax
        self.save_head_reduced_attn = save_head_reduced_attn
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
        logging.getLogger(self.__class__.__name__).debug(
            f'Update the temperature from {self.tau} to {tau}'
        )
        self.tau = tau

    def set_head_mask(self, head_graph_mask: torch.LongTensor) -> None:
        """
        Since the Attention supports the multihead feature, it's possible to set graph masks by heads.
        :param head_graph_mask: (batch, head, M, N), to inject connections for individual heads.
        """
        self._head_graph_mask = head_graph_mask

    def clear_head_mask(self):
        self._head_graph_mask = None

    def add_runtime_weight_hooks(self, hook: TENSOR_FUNC) -> None:
        self._runtime_weight_hooks.append(hook)

    def clear_runtime_weight_hooks(self):
        self._runtime_weight_hooks = []

    def forward(self,
                inputs: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: torch.LongTensor | None = None,
                graph_mask: torch.LongTensor | None = None,
                ) -> torch.Tensor:
        """
        :param inputs: (batch, M, input_dim) or simply (batch, input_dim)
        :param attend_over: (batch, N, attend_dim)
        :param attend_mask: (batch, N)
        :param graph_mask: (batch, M, N)
        """
        bsz = inputs.size(0)

        # (bsz * head, [qkv]_length, d_[qkv])
        query, key, value = self._pre_attn_process(inputs, attend_over)
        value = value.view(bsz, self.n_head, *value.size()[-2:])

        attn = self.attn_scorer(query, key)    # (b * head, M, N)
        attn_logit = attn.view(bsz, self.n_head, *attn.size()[-2:]) # (b, head, M, N)
        attn_prob = self._compute_weight(attn_logit, attend_mask)   # (b, head, M, N)
        self._save_weight(attn_prob, inputs.ndim < 3)
        attn_prob = self._call_weight_hooks(attn_prob)

        # ctx: (b, M, d)
        ctx = self._ctx_delegation(attn_prob, value)
        ctx = self.post_ctx_mapping(ctx)
        if inputs.ndim < 3:     # single token attention, remove the M=1 dim
            ctx = ctx.squeeze(-2)
        return ctx

    def _pre_attn_process(self, inputs, attend_over):
        assert inputs.ndim in (2, 3)

        # pre-attn mappings
        query = self.pre_q_mapping(inputs)          # (b, [M,], d_q)
        key = self.pre_k_mapping(attend_over)       # (b, N, d_k)
        value = self.pre_v_mapping(attend_over)     # (b, N, d_v)

        def _split_heads(t: torch.Tensor):
            if t.ndim < 3:
                t = t.unsqueeze(1)  # new axis after the batch dim
            bsz, t_len, hid = t.size()
            assert hid % self.n_head == 0
            head_t = t.reshape(bsz, t_len, self.n_head, hid // self.n_head)
            head_t = head_t.transpose(1, 2).reshape(bsz * self.n_head, t_len, hid // self.n_head)
            return head_t

        # processing multiheads
        headed_query, headed_key, headed_val = map(_split_heads, (query, key, value))
        return headed_query, headed_key, headed_val

    def _compute_weight(self, attn_logit, attn_mask=None, graph_mask=None):
        """
        Compute the attention probabilities from logits, given various masks.

        :param attn_logit: (batch, head, M, N)
        :param attn_mask: (batch, N), used to wipe out paddings in the attn targets.
        :param graph_mask: (batch, M, N), to inject graph connections of attention inputs ant targets.
        :return: attn_prob: (batch, head, M, N), the last dim sums to 1
        """
        bsz, nhead, inp_len, attn_len = attn_logit.size()
        mask = torch.ones_like(attn_logit, dtype=torch.long)
        if attn_mask is not None:
            mask *= attn_mask.view(bsz, 1, 1, attn_len)
        if graph_mask is not None:
            mask *= graph_mask.view(bsz, 1, inp_len, attn_len)
        if self._head_graph_mask is not None:
            mask *= self._head_graph_mask

        if self.tau_is_fixed:
            temperature = self.init_tau
        else:
            temperature = max(self.tau, self.min_tau) if self.training else self.min_tau

        if self.is_sparse:
            attn_prob = masked_sparsemax(attn_logit / temperature, mask.bool())
        else:
            attn_prob = masked_softmax(attn_logit / temperature, mask.bool())
        return attn_prob

    def _save_weight(self, attn_prob, is_token_attn):
        if is_token_attn:
            # remove the M=1 dimension if required
            attn_prob = attn_prob.mean(dim=-2)
        if self.save_head_reduced_attn:
            attn_prob = attn_prob.mean(dim=1)
        self.save_last_attn_weights(attn_prob)

    def _call_weight_hooks(self, attn_prob: torch.Tensor) -> torch.Tensor:
        for hook in self._runtime_weight_hooks:
            attn_prob = hook(attn_prob)

        return attn_prob

    def _ctx_delegation(self, head_weights, value):
        """
        :param head_weights: (batch, head, M, N)
        :param value: (batch, head, N, d_v)
        :return: (batch, M, d), d == d_v * head
        """
        method = self.training_ctx if self.training else self.eval_ctx
        if method == 'weighted_sum':
            return _ctx_by_weighted_sum(head_weights, value)
        elif method == 'argmax':
            return _ctx_by_argmax(head_weights, value)
        elif match := re.match(r'top(\d+)_norm', method):
            return _ctx_by_normed_topk(head_weights, value, num_k=int(match.group(1)))
        else:
            raise ValueError(f'the specified context delegation {method} '
                             f'for {"training" if self.training else "evaluation"} '
                             f'is unknown.')


def get_attention(attn_type: str,
                  vector_dim: int = 0,
                  matrix_dim: int = 0,
                  **kwargs):
    """
    Build an Attention module with specified parameters.
    :param attn_type: indicates the attention type, e.g. "bilinear", "dot_product" or "none"
    :param vector_dim: the vector to compute attention
    :param matrix_dim: the bunch of vectors to be attended against (batch, num, matrix_dim)
    :return: a torch.nn.Module
    """

    attn_type = attn_type.lower()
    if attn_type in ("bilinear", "dot_product"):
        attn_type = 'adaptive_' + attn_type

    if attn_type == "generalized_bilinear":
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

    elif attn_type == 'adaptive_dot_product':
        from allennlp.modules.matrix_attention import DotProductMatrixAttention
        attn = AdaptiveGeneralAttention(AdaptiveAllenLogits(DotProductMatrixAttention()))

    elif attn_type == 'adaptive_bilinear':
        from allennlp.modules.matrix_attention import BilinearMatrixAttention
        attn = AdaptiveGeneralAttention(AdaptiveAllenLogits(BilinearMatrixAttention(vector_dim, matrix_dim)))

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

