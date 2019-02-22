from typing import List, Mapping, Dict, Optional, Tuple, Union, Callable, Sequence
import torch
import torch.nn

from utils.nn import add_depth_features_to_single_position, AllenNLPAttentionWrapper, select_item_along_dim
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper
from models.universal_hidden_state_wrapper import UniversalHiddenStateWrapper

class DepthEmbeddingType:
    SINUSOID = "sinusoid"
    LEARNT = "learnt"
    NONE = "none"

class AdaptiveStateMode:
    BASIC = "basic"
    RANDOM = "random"
    MEAN_FIELD = "mean_field"

class AdaptiveRNNCell(torch.nn.Module):
    def forward(self, inputs, hidden, enc_attn_fn) -> Tuple[torch.Tensor, torch.Tensor,
                                                            Optional[torch.Tensor], Optional[torch.Tensor]]:
        raise NotImplementedError

    def init_hidden_states(self, source_state, source_mask, is_bidirectional=False):
        """
        Init the hidden states for decoder.

        :param source_state: encoded hidden states at input
        :param source_mask: padding mask for inputs
        :param is_bidirectional:
        :return: (hidden, output) or ( (hidden, context), output) for LSTM
        """
        raise NotImplementedError


class ACTRNNCell(AdaptiveRNNCell):
    """
    An RNN-based cell, which adaptively computing the hidden states along depth dimension.
    """
    def __init__(self,
                 hidden_dim: int,
                 rnn_cell: UniversalHiddenStateWrapper,
                 use_act: bool = True,
                 act_dropout: float = .1,
                 act_max_layer: int = 10,
                 act_epsilon: float = .1,
                 depth_wise_attention: Union[AllenNLPAttentionWrapper, SingleTokenMHAttentionWrapper, None] = None,
                 depth_embedding_type: str = DepthEmbeddingType.SINUSOID,
                 state_mode: str = AdaptiveStateMode.BASIC,
                 ):
        """
        :param hidden_dim: dimension of input, output and attention embedding
        :param use_act: use adaptive computing
        :param act_dropout: dropout ratio in halting network
        :param act_max_layer: the maximum number of layers for adaptive computing
        :param act_epsilon: float in (0, 1), a reserved range of probability to halt
        :param depth_wise_attention: do input attention over the adaptive RNN output
        :param depth_embedding_type: add some timestep to the SINUSOID
        :param rnn_cell: some basic RNN cell, e.g., vanilla RNN, LSTM, GRU, accepts inputs across depth
        """
        super(ACTRNNCell, self).__init__()

        self._rnn_cell: UniversalHiddenStateWrapper = rnn_cell
        self._use_act = use_act

        self._halting_fn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(act_dropout),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

        self._threshold = 1 - act_epsilon
        self._max_computing_time = act_max_layer

        self._depth_wise_attention = depth_wise_attention
        self._dwa_mapping = torch.nn.Linear(hidden_dim, hidden_dim)
        self._depth_embedding_type = depth_embedding_type
        if self._depth_embedding_type == DepthEmbeddingType.LEARNT:
            self._depth_embedding = torch.nn.Parameter(torch.Tensor(act_max_layer, hidden_dim))

        self.state_mode = state_mode
        self.hidden_dim = hidden_dim

    def forward(self, inputs: torch.Tensor, hidden, enc_attn_fn: Optional[Callable]):
        """
        :param inputs: (batch, hidden_dim)
        :param hidden: some hidden states with unknown internals
        :param enc_attn_fn: Attention function, in case or the adaptively computed states need attention manipulation
                The function maps the state directly to a context vector:
                i.e. (batch, hidden_dim) -> (batch, hidden_dim)
        :return: (batch, hidden_dim) or [(batch, hidden_dim)]
        """
        if not self._use_act:
            hidden, out = self._rnn_cell(inputs, hidden)
            return hidden, out, None, None

        else:
            return self._forward_act(inputs, hidden, enc_attn_fn)


    def _forward_act(self, inputs: torch.Tensor, hidden, enc_attn_fn):
        """
        :param inputs: (batch, hidden_dim)
        :param hidden: some hidden state recognizable by the universal RNN cell wrapper
        :param enc_attn_fn: Attention function, in case or the adaptively computed states need attention manipulation
                The function maps the state directly to a context vector:
                i.e. (batch, hidden_dim) -> (batch, hidden_dim)
        :return: (batch, hidden_dim) or [(batch, hidden_dim)]
        """
        depth = 0
        output = self._rnn_cell.get_output_state(hidden) # output initialized as in the last time

        # halting_prob_cumulation: (batch,)
        # halting_prob_list: [(batch,)]
        # hidden_list: [hidden]
        # output_list: [(batch, hidden_dim)]
        # alive_mask_list: [(batch,)]
        batch = inputs.size()[0]
        halting_prob_cumulation = inputs.new_zeros(batch).float()
        halting_prob_list = []
        hidden_list = []
        output_list = []
        alive_mask_list = []

        while depth < self._max_computing_time and (halting_prob_cumulation < 1.).any():
            # current all alive tokens, which need further computation
            # alive_mask: (batch,)
            alive_mask: torch.Tensor = halting_prob_cumulation < 1.
            alive_mask = alive_mask.float()

            # halting_prob: (batch, ) <- (batch, 1)
            context = enc_attn_fn(output) if enc_attn_fn is not None else 0
            halting_prob = self._halting_fn(output + context).squeeze(-1)

            # mask to the newly halted tokens, which is exhausted at the current timestep of computation.
            # if the depth hits its upper bound, all nodes should be halted
            # new_halted: (batch,)
            new_halted = ((halting_prob * alive_mask + halting_prob_cumulation) > self._threshold).float()
            if depth == self._max_computing_time - 1:
                new_halted = new_halted.new_ones(batch).float()
            remainder = 1. - halting_prob_cumulation + 1.e-15

            # all tokens that survives from the current timestep's computation
            # alive_mask: (batch, )
            alive_mask = (1 - new_halted) * alive_mask

            # cumulations for newly halted positions will reach 1.0 after adding up remainder at the current timestep
            halting_prob_per_step = halting_prob * alive_mask + remainder * new_halted
            halting_prob_cumulation = halting_prob_cumulation + halting_prob_per_step

            # Every hidden state at present is paired with the alive mask telling
            # which states needs further computation.
            # And the halting probability accompanied at the step is either the halting probability itself
            # if it's not the last computation step of the very token, or the remainder for tokens which
            # would be halted at this step.
            hidden_list.append(hidden)
            output_list.append(output)
            alive_mask_list.append(alive_mask)
            halting_prob_list.append(halting_prob_per_step)

            # step_inputs: (batch, hidden_dim)
            step_inputs = inputs
            step_inputs = self._add_depth_embedding(step_inputs, depth)
            step_inputs = self._add_depth_wise_attention(step_inputs, output, output_list)
            hidden, output = self._rnn_cell(step_inputs, hidden)
            depth += 1

        # halting_probs: (batch, max_computing_depth)
        # alive_masks: (batch, max_computing_depth)
        halting_probs = torch.stack(halting_prob_list, dim=-1).float()
        alive_masks = torch.stack(alive_mask_list, dim=-1).float()
        merged_hidden = self._adaptively_merge_hidden_list(hidden_list, halting_probs, alive_masks)
        merged_output = self._rnn_cell.get_output_state(merged_hidden)

        accumulated_halting_probs = (halting_probs * alive_masks).sum(1)
        num_updated = alive_masks.sum(1)

        return merged_hidden, merged_output, accumulated_halting_probs, num_updated

    def _add_depth_embedding(self, inputs: torch.Tensor, depth: int) -> torch.Tensor:
        if depth == 0:
            return inputs

        if self._depth_embedding_type == DepthEmbeddingType.SINUSOID:
            step_inputs = add_depth_features_to_single_position(inputs, depth)

        elif self._depth_embedding_type == DepthEmbeddingType.LEARNT:
            # every item in batch receives the same depth embedding
            step_inputs = inputs + self._depth_embedding.select(0, depth).unsqueeze(0)

        else: #self._depth_embedding_type == DepthEmbeddingType.NONE:
            step_inputs = inputs

        return step_inputs

    def _adaptively_merge_hidden_list(self, hidden_list, halting_probs, alive_masks):
        # halting_probs: (batch, max_computing_depth)
        # alive_masks: (batch, max_computing_depth)
        batch, max_depth = halting_probs.size()

        if self.state_mode == AdaptiveStateMode.BASIC:
            # desired_state: (batch, hidden_dim)
            weight = torch.zeros_like(halting_probs)
            weight[torch.arange(batch), alive_masks.sum(1).long()] = 1

        elif self.state_mode == AdaptiveStateMode.MEAN_FIELD:
            # halting_probs_extended: (batch, 1, max_computing_depth)
            # desired_state: (batch, hidden_dim)
            weight = halting_probs

        elif self.state_mode == AdaptiveStateMode.RANDOM:
            # samples_index: torch.LongTensor: (batch, )
            # desired_state: (batch, hidden_dim)
            samples_index = torch.multinomial(halting_probs, 1).squeeze(-1)
            weight = torch.zeros_like(halting_probs)
            weight[torch.arange(batch), samples_index] = 1

        else:
            raise ValueError(f'Adaptive State Mode {self.state_mode} not supported')

        merged_hidden = self._rnn_cell.merge_hidden_list(hidden_list, weight)

        return merged_hidden

    def _add_depth_wise_attention(self,
                                  step_inputs: torch.Tensor,
                                  output: torch.Tensor,
                                  previous_output_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Get depth-wise attention over the previous hiddens

        :param step_inputs: (batch, hidden_dim)
        :param output: (batch, hidden_dim)
        :param previous_output_list: [(batch, hidden_dim)]
        :return: (batch, hidden_dim)
        """
        if self._depth_wise_attention is None:
            return step_inputs

        # attend_over: (batch, steps, hidden_dim)
        attend_over = torch.stack(previous_output_list, dim=1)

        # context: (batch, hidden_dim)
        context = self._depth_wise_attention(output, attend_over)
        context = self._dwa_mapping(context)

        step_inputs = step_inputs + context

        return step_inputs

    def init_hidden_states(self, source_state, source_mask, is_bidirectional=False):
        return self._rnn_cell.init_hidden_states(source_state, source_mask, is_bidirectional)

