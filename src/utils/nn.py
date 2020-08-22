from typing import List, Tuple, Dict, Mapping, Optional
import torch
import torch.nn
import re

from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.modules.attention import Attention
from allennlp.nn.util import masked_softmax

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

    :param inputs: (batch, emb_dim)
    :param timestep: float
    :returns same shape with inputs: (batch, emb_dim)
    """
    return add_position_and_timestep_sinusoid(inputs.unsqueeze(1), timestep).squeeze(1)

class AllenNLPMatrixAttentionWrapper(torch.nn.Module):
    """
    A wrapper for matrix attention in allennlp, fitting the interface of the multi-headed attention
    defined in models.transformer.multi_head_attention
    """
    def __init__(self, attn: MatrixAttention):
        super(AllenNLPMatrixAttentionWrapper, self).__init__()
        self._attn: MatrixAttention = attn

    def forward(self,
                input: torch.Tensor,
                attend_over: torch.Tensor,
                attend_mask: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrap the Attention in AllenNLP, with sufficient dimension and context value computation

        :param input: (batch, max_input_length, input_dim)
        :param attend_over: (batch, max_attend_length, attend_dim)
        :param attend_mask: (batch, max_attend_length), used to blind out padded tokens
        :return: Tuple of context vector and attention vector:
                   context: (batch, max_input_length, output_dim=attend_dim)
                 attention: (batch, max_input_length, 1, max_attend_length)
        """

        # logits: (batch, max_input_length, max_attend_length)
        logits = self._attn(input, attend_over)

        # attend_mask: (batch, 1, max_attend_length)
        attend_mask = attend_mask.unsqueeze(1)

        # attn: (batch, max_input_length, max_attend_length)
        attn = masked_softmax(logits, attend_mask)

        # context: (batch, max_input_length, attend_dim)
        context = torch.matmul(attn, attend_over)

        return context, attn.unsqueeze(-2)


class AllenNLPAttentionWrapper(torch.nn.Module):
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

def filter_cat(iterable, dim):
    items = [item for item in iterable if item is not None]
    res = torch.cat(items, dim=dim)
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

def seq_likelihood(logits: torch.FloatTensor,
                   targets: torch.LongTensor,
                   weights: torch.FloatTensor,
                   ):
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    probs_flat = torch.nn.functional.softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    # Contribution to the negative log likelihood only comes from the exact indices
    # of the targets, as the target distributions are one-hot. Here we use torch.gather
    # to extract the indices of the num_classes dimension which contribute to the loss.
    # shape : (batch * sequence_length, 1)
    likelihood_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    likelihood = likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    likelihood = likelihood * weights.float()

    # shape : (batch_size,)
    per_batch_loss = likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
    return per_batch_loss


def seq_cross_ent(logits: torch.FloatTensor,
                  targets: torch.LongTensor,
                  weights: torch.FloatTensor,
                  average: Optional[str] = "batch",
                  ):
    """
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)

    Returns
    -------
    A torch.FloatTensor representing the scalar cross entropy loss.
    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    # Contribution to the negative log likelihood only comes from the exact indices
    # of the targets, as the target distributions are one-hot. Here we use torch.gather
    # to extract the indices of the num_classes dimension which contribute to the loss.
    # shape : (batch * sequence_length, 1)
    negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss


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

def get_decoder_initial_states(layer_state: List[torch.Tensor],
                               source_mask: torch.LongTensor,
                               strategy: str = "forward_last_all",
                               is_bidirectional: bool = False,
                               num_decoder_layers: int = 1,
                               ) -> List[torch.Tensor]:
    """
    Initialize the hidden states for decoder given encoders
    :param layer_state: [(batch, src_len, hidden_dim)]
    :param source_mask: (batch, src_len)
    :param strategy: a string to indicate how to aggregate and assign initial states to the decoder
                    typically available strategies:
                    [avg|pooling|last|zero]_[lowest|all|parallel]
                    which means
                    1) to use some aggregation heuristics for the encoder, and
                    2) apply to the decoder initial hidden states
    :param is_bidirectional: if the encoder output is a concatenated vector of bidirectional outputs
    :param num_decoder_layers: how many stacked layers does the decoder have
    :return: a list of selected states of each layer to be assigned to the decoder
    """
    m = re.match(r"(avg|max|forward_last|zero)_(lowest|all|parallel)", strategy)
    if not m:
        raise ValueError(f"specified strategy '{strategy}' not supported")

    agg_stg, assign_stg = m.group(1), m.group(2)
    batch, _, hidden_dim = layer_state[0].size()
    source_mask_expand = source_mask.unsqueeze(-1).float() # (batch, seq_len, hidden)
    if agg_stg == "avg":
        src_agg = [
            (l * source_mask_expand).sum(1) / (source_mask_expand.sum(1) + 1e-30)
            for l in layer_state
        ]

    elif agg_stg == "max":
        src_agg = [
            ((source_mask_expand + 1e-45).log() + l).max(1)
            for l in layer_state
        ]
    elif agg_stg == "zero":
        src_agg = [source_mask.new_zeros((batch, hidden_dim), dtype=torch.float32) for _ in layer_state]
    else: # forward_last
        # last_word_indices: (batch,)
        # expanded_indices: (batch, 1, hidden_dim)
        # forward_by_layer: [(batch, hidden_dim)]
        last_word_indices = source_mask.sum(1).long() - 1
        expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch, 1, hidden_dim)
        forward_by_layer = [state.gather(1, expanded_indices).squeeze(1) for state in layer_state]
        if is_bidirectional:
            hidden_dim = hidden_dim // 2
            src_agg = [state[:, :hidden_dim] for state in forward_by_layer]
        else:
            src_agg = forward_by_layer

    if assign_stg == "lowest": # use the top layer aggregated state for the decoder bottom, zero for others
        decoder_hidden = [src_agg[-1]] + [source_mask.new_zeros((batch, hidden_dim), dtype=torch.float32)
                                          for _ in range(num_decoder_layers - 1)]
    elif assign_stg == "all": # use the same top layer state for all decoder layers
        decoder_hidden = [src_agg[-1] for _ in range(num_decoder_layers)]
    else: # parallel, each encoder is used for the appropriate decoder layer
        decoder_hidden = src_agg

    return decoder_hidden
