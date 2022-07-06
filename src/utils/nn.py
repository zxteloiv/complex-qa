from typing import List, Tuple, Dict, Mapping, Optional, Literal
import torch
import torch.nn
import torch.nn.functional
import re
import math


def add_position_and_timestep_sinusoid(inputs: torch.Tensor,
                                       timestep: Optional[float] = None,
                                       base_num: float = 1.e4) -> torch.Tensor:
    """
    Add positional embedding to inputs, which contains words at every position

    :param inputs: input embedding for entire sequence, a shape of (batch, seq_len, embedding) assumed
    :param timestep: which timestep is the current recurrent block

    :param base_num: the base number used in dimensional power, default to 10^4 as the Attention is All You Need paper.

    :return: same shape as inputs
    """
    _, seq_len, emb_dim = inputs.size()
    assert emb_dim % 2 == 0, "embedding dimension must be even"

    # position: [0, 1, ..., seq_len - 1] with shape (seq_len, )
    # negative_half_dim_index: [-0, -1, ..., -(emb_dim // 2 - 1)] with shape (emb_dim // 2, )
    # inverse_dim: [base_num ^(-0), ..., base_num^(-(emb_dim // 2 - 1))], with shape (emb_dim // 2,)
    position: torch.Tensor = inputs.new_ones(seq_len).cumsum(dim=0).float() - 1
    negative_half_dim_index: torch.Tensor = -(inputs.new_ones(emb_dim // 2).cumsum(dim=0).float() - 1)
    inverse_dim: torch.Tensor = torch.pow(base_num, negative_half_dim_index)

    # x: (seq_len, emb_dim // 2) <- (seq_len, 1) * (1, emb_dim // 2)
    x: torch.Tensor = position.unsqueeze(1) * inverse_dim.unsqueeze(0)

    if timestep is not None:
        # y: (1, emb_dim // 2)
        y: torch.Tensor = inverse_dim.unsqueeze(0) * timestep
        sinusoid_odd, sinusoid_even = x.sin() + y.sin(), x.cos() + y.cos()
    else:
        sinusoid_odd, sinusoid_even = x.sin(), x.cos()

    # sinusoid: (seq_len, emb_dim // 2, 2) -> (1, seq_len, emb_dim)
    sinusoid: torch.Tensor = torch.stack([sinusoid_odd, sinusoid_even], dim=2).reshape(1, seq_len, -1)

    return inputs + sinusoid


def add_positional_features(inputs: torch.Tensor) -> torch.Tensor:
    """
    A wrapper with the same name from AllenNLP
    """
    return add_position_and_timestep_sinusoid(inputs, None)


def add_depth_features_to_single_position(inputs: torch.Tensor, timestep: float) -> torch.Tensor:
    """
    Add depth-wise features to inputs.
    The word ``depth'' is similar to ``timestep'' in Transformer.

    :param inputs: (batch, seq_len, emb_dim)
    :param timestep: float
    :returns same shape with inputs: (batch, seq_len, emb_dim)
    """
    return add_position_and_timestep_sinusoid(inputs, timestep)


def filter_cat(iterable, dim):
    items = [item for item in iterable if item is not None]
    res = torch.cat(items, dim=dim) if len(items) > 0 else None
    return res


def filter_sum(iterable):
    items = [item for item in iterable if item is not None]
    res = None
    for item in items:
        if res is None:
            res = item
        else:
            res = res + item
    return res


def roll(t: torch.Tensor, shift: int, axis: int = 0):
    length = t.size()[axis]
    assert length > 1, "The data size must be greater than 1, otherwise rolling is "
    shift = shift % length
    chunks = torch.split(t, [length - shift, shift], dim=axis)
    return torch.cat(list(reversed(chunks)), dim=axis)


def prepare_input_mask(tokens, padding_val: int = 0):
    if tokens is not None:
        # padding token is assigned 0 in NSVocab by default
        token_ids, mask = tokens, (tokens != padding_val).long()
    else:
        token_ids, mask = None, None
    return token_ids, mask


def logits_to_prob(logits: torch.FloatTensor, log_transform: Literal['none', 'bounded', 'raw'] = 'none') -> torch.Tensor:
    # probs: (batch, ..., num_classes)
    if log_transform == 'none':
        return torch.nn.functional.softmax(logits, dim=-1)

    elif log_transform == 'bounded':
        probs = torch.nn.functional.softmax(logits, dim=-1) + 1
        log_probs = probs.log()
    else:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    return log_probs


def masked_reducing_gather(source: torch.Tensor,
                           targets: torch.LongTensor,
                           weights: Optional[torch.Tensor],
                           reducing_method: Literal["token", "batch", "none"]):
    """
    :param source: (..., dim)
    :param targets: (...,) -> (..., 1)
    :param weights: (...,)
    :param reducing_method: reduction methods after gathering
    """
    if reducing_method not in {None, "token", "batch", "none"}:
        raise ValueError(f"Got method {reducing_method}, expected one of None, 'token', or 'batch'")

    assert targets.ndim == weights.ndim
    if targets.ndim < source.ndim:
        targets = targets.unsqueeze(-1)

    output = torch.gather(source, dim=-1, index=targets).squeeze(-1)
    if weights is not None:
        output = output * weights.float()

    # (batch,)
    batch_reduction = sum_to_batch_size(output)
    weights = torch.ones_like(batch_reduction) if weights is None else sum_to_batch_size(weights)

    if reducing_method == 'batch':
        per_batch_loss = batch_reduction / (weights + 1e-13)
        num_non_empty_sequences = ((weights > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences           # scalar
    elif reducing_method == "token":
        return batch_reduction.sum() / (weights.sum().float() + 1e-13)  # scalar
    else:
        return batch_reduction / (weights + 1e-13)  # (batch,)


def seq_likelihood(logits: torch.FloatTensor, targets: torch.LongTensor, weights: torch.FloatTensor):
    probs = logits_to_prob(logits, log_transform='none')
    ll = masked_reducing_gather(probs, targets, weights, reducing_method='none')
    return ll


def seq_cross_ent(logits: torch.FloatTensor,
                  targets: torch.LongTensor,
                  weights: torch.FloatTensor,
                  average: Literal["token", "batch", "none"] = "batch"):
    """
    :param logits: (batch_size, ..., num_classes), the logit (unnormalized probability) for each class.
    :param targets: (batch, ...) the index of the true class for each corresponding step.
    :param weights: (batch, ...)
    :param average: reduction method
    :return (batch, ) if average mode is none or (0,) if average mode is token or batch
    """
    log_probs = logits_to_prob(logits, log_transform='raw')
    nll = - masked_reducing_gather(log_probs, targets, weights, reducing_method=average)
    return nll


def sum_to_batch_size(t: torch.Tensor):
    reducible_dims = list(range(t.ndim))[1:]
    return t.sum(reducible_dims)


def seq_masked_mean(x: torch.Tensor, mask: torch.LongTensor, dim: int = 1):
    """
    estimate the mean of given sequence x along the length dimension,
    not being concerned with masked slots as speicified.
    :param x: (batch, L, dim)
    :param mask: (batch, L)
    :return: mean of x: (batch, dim)
    """
    summation: torch.Tensor = (x * mask.unsqueeze(-1)).sum(dim=dim)  # (batch, dim)
    count: torch.Tensor = mask.sum(-1, keepdim=True).float() # (batch, 1)
    mean = summation / (count + 1e-20)
    return mean


def seq_masked_var_mean(x: torch.Tensor, mask: torch.LongTensor, dim: int = 1):
    """
    estimate the variance and mean of given sequence x along the length dimension,
    not being concerned with masked slots as speicified.
    :param x: (batch, L, dim)
    :param mask: (batch, L)
    :return: mean of x: (batch, dim)
    """
    summation: torch.Tensor = (x * mask.unsqueeze(-1)).sum(dim=dim)  # (batch, dim)
    squared_summation: torch.Tensor = ((x * mask.unsqueeze(-1)) ** 2).sum(dim=dim)  # (batch, dim)

    count: torch.Tensor = mask.sum(-1, keepdim=True).float() # (batch, 1)
    mean = summation * (count + 1e-13).reciprocal() # (batch, dim)

    var = squared_summation - mean ** 2
    return var, mean


def seq_masked_std_mean(x: torch.Tensor, mask: torch.LongTensor, dim: int = 1):
    """
    estimate the standard variance and mean of given sequence x along the length dimension,
    not being concerned with masked slots as speicified.
    :param x: (batch, L, dim)
    :param mask: (batch, L)
    :return: mean of x: (batch, dim)
    """
    var, mean = seq_masked_var_mean(x, mask, dim)
    std = (var.abs() + 1e-20) ** 0.5
    return std, mean


def get_final_encoder_states(encoder_outputs: torch.Tensor,
                             mask: torch.Tensor,
                             bidirectional: bool = False) -> torch.Tensor:
    """
    Given the output from a ``Seq2SeqEncoder``, with shape ``(batch_size, sequence_length,
    encoding_dim)``, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape ``(batch_size, encoding_dim)``.  This is not as simple as
    ``encoder_outputs[:, -1]``, because the sequences could have different lengths.  We use the
    mask (which has shape ``(batch_size, sequence_length)``) to find the final state for each batch
    instance.

    If all words are masked, which is an edge case not considered in the original AllenNLP code,
    we simply chose the first (id=0) output as the embedding.

    Additionally, if ``bidirectional`` is ``True``, we will split the final dimension of the
    ``encoder_outputs`` into two and assume that the first half is for the forward direction of the
    encoder and the second half is for the backward direction.  We will concatenate the last state
    for each encoder dimension, giving ``encoder_outputs[:, -1, :encoding_dim/2]`` concatenated with
    ``encoder_outputs[:, 0, encoding_dim/2:]``.
    """
    # These are the indices of the last words in the sequences (i.e. length sans padding - 1).  We
    # are assuming sequences are right padded.
    # Shape: (batch_size,)
    last_word_indices = mask.sum(1).long() - 1
    last_word_indices = last_word_indices * (last_word_indices >= 0)
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    # Shape: (batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2):]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def aggregate_layered_state(layer_state: List[torch.Tensor],
                            source_mask: torch.LongTensor,
                            strategy: str = "forward_last_all",
                            use_first_half: bool = False,
                            ) -> List[torch.Tensor]:
    """
    Initialize the states for stacked decoder given stacked encoders.
    The output is usually passed further to the stacked decoder to initialize the blackbox hidden states.
    :param layer_state: [(batch, src_len, hidden_dim)]
    :param source_mask: (batch, src_len)
    :param strategy: a string to indicate how to aggregate and assign initial states to the decoder
                    typically available strategies:
                    [avg|max|forward_last|zero]_[lowest|all|parallel]
                    which means
                    1) to use some aggregation heuristics for the encoder, and
                    2) apply to the decoder initial hidden states
    :param use_first_half: use the first half if the encoder output is a concatenated vector of bidirectional outputs
    :return: a list of selected states of each layer to be assigned to the decoder
    """
    m = re.match(r"(avg|max|forward_last|zero)_?(lowest|all|parallel)?", strategy)
    if not m:
        raise ValueError(f"specified strategy '{strategy}' not supported")
    agg_stg, assign_stg = m.group(1), m.group(2)

    batch, _, hidden_dim = layer_state[0].size()
    source_mask_expand = source_mask.unsqueeze(-1).float() # (batch, seq_len, hidden)
    if agg_stg == "avg":
        return [(state * source_mask_expand).sum(1) / (source_mask_expand.sum(1) + 1e-30) for state in layer_state]
    if agg_stg == "max":
        return [((source_mask_expand + 1e-45).log() + state).max(1) for state in layer_state]
    if agg_stg == "zero":
        return [source_mask.new_zeros((batch, hidden_dim), dtype=torch.float32) for _ in layer_state]

    # forward_last
    # last_word_indices: (batch,)
    last_word_indices = (source_mask.sum(1).long() - 1)
    # for the case the entire sequence is masked out, the last indices is set to 0 by default,
    # because they don't contribute to the final loss.
    last_word_indices = last_word_indices * (last_word_indices >= 0)
    # expanded_indices: (batch, 1, hidden_dim)
    # forward_by_layer: [(batch, hidden_dim)]
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch, 1, hidden_dim)
    forward_by_layer = [state.gather(1, expanded_indices).squeeze(1) for state in layer_state]
    if use_first_half:
        hidden_dim = hidden_dim // 2
        src_agg = [state[:, :hidden_dim] for state in forward_by_layer]
    else:
        src_agg = forward_by_layer

    return src_agg


def assign_stacked_states(src_agg: List[torch.Tensor], num_layers: int, strategy: str = "forward_last_all",):
    """
    Usually used to built stacked decoder states from aggregated outputs of a stacked encoder
    :param src_agg:
    :param num_layers:
    :param strategy:
    :return:
    """
    m = re.match(r"(avg|max|forward_last|zero)?_?(lowest|all|parallel)", strategy)
    if not m:
        raise ValueError(f"specified strategy '{strategy}' not supported")
    policy = m.group(2)
    if policy == "lowest": # use the top layer aggregated state for the decoder bottom, zero for others
        init_state = [src_agg[-1]] + [torch.zeros_like(src_agg[-1]) for _ in range(num_layers - 1)]

    elif policy == "all": # use the same top layer state for all decoder layers
        init_state = [src_agg[-1] for _ in range(num_layers)]

    else:   # parallel, each encoder is used for the appropriate decoder layer
        assert len(src_agg) == num_layers
        init_state = src_agg

    return init_state


def expand_tensor_size_at_dim(t: torch.Tensor, size: int, dim: int =-2) -> torch.Tensor:
    old_size = t.size()
    t = t.unsqueeze(dim=dim)
    if dim == -1:
        return t.expand(*old_size, size)
    elif dim < 0:
        return t.expand(*old_size[:dim + 1], size, *old_size[dim + 1:])
    else:
        return t.expand(*old_size[:dim], size, *old_size[dim:])
