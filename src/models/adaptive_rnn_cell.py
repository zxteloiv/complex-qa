from typing import List, Mapping, Dict, Optional, Tuple, Union, Callable, Sequence
import torch
import torch.nn

from utils.nn import add_depth_features_to_single_position, AllenNLPAttentionWrapper, select_item_along_dim
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper

class DepthEmbeddingType:
    SINUSOID = "sinusoid"
    LEARNT = "learnt"
    NONE = "none"

class AdaptiveStateMode:
    BASIC = "basic"
    RANDOM = "random"
    MEAN_FIELD = "mean_field"

class RNNType:
    VanillaRNN = torch.nn.RNNCell
    LSTM = torch.nn.LSTMCell
    GRU = torch.nn.GRUCell

class AdaptiveRNNCell(torch.nn.Module):
    """
    An RNN-based cell, which adaptively computing the hidden states along depth dimension.
    """
    def __init__(self,
                 hidden_dim: int,
                 rnn_cell: torch.nn.Module,
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
        super(AdaptiveRNNCell, self).__init__()

        self._rnn_type = type(rnn_cell)
        self._rnn_cell = rnn_cell
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
        self._depth_embedding_type = depth_embedding_type
        if self._depth_embedding_type == DepthEmbeddingType.LEARNT:
            self._depth_embedding = torch.nn.Parameter(torch.Tensor(act_max_layer, hidden_dim))

        self.state_mode = state_mode
        self.hidden_dim = hidden_dim

    def forward(self,
                inputs: torch.Tensor,
                hidden: Union[Sequence, torch.Tensor, None] = None,
                ) -> Union[Sequence, torch.Tensor]:
        """

        :param inputs: (batch, hidden_dim)
        :param hidden: (batch, hidden_dim) or [(batch, hidden_dim)]
        :return: (batch, hidden_dim) or [(batch, hidden_dim)]
        """
        if not self._use_act:
            res = self._rnn_cell(inputs, hidden)
            return res, None, None

        depth = 0
        # halting_prob_cumulation: (batch,)
        # halting_prob_list: [(batch,)]
        # hidden_list: [(batch, hidden_dim)] or [Tuple[(batch, hidden_dim), (batch, hidden_dim)]]
        # alive_mask_list: [(batch,)]
        batch = inputs.size()[0]
        halting_prob_cumulation = inputs.new_zeros(batch).float()
        halting_prob_list = []
        hidden_list = []
        alive_mask_list = []

        while depth < self._max_computing_time and (halting_prob_cumulation < 1.).any():
            # current all alive tokens, which need further computation
            # alive_mask: (batch,)
            alive_mask: torch.Tensor = halting_prob_cumulation < 1.
            alive_mask = alive_mask.float()

            # halting_prob: (batch, ) <- (batch, 1)
            halting_prob = self._halting_fn(self.get_output_state(hidden)).squeeze(-1)

            # mask to the newly halted tokens, which is exhausted at the current timestep of computation
            # new_halted: (batch,)
            new_halted = ((halting_prob * alive_mask + halting_prob_cumulation) > self._threshold).float()
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
            alive_mask_list.append(alive_mask)
            halting_prob_list.append(halting_prob_per_step)

            # step_inputs: (batch, hidden_dim)
            step_inputs = inputs
            step_inputs = self._add_depth_wise_attention(step_inputs, hidden, hidden_list)
            step_inputs = self._add_depth_embedding(step_inputs, depth)
            hidden = self._rnn_cell(step_inputs, hidden)
            depth += 1

        # halting_probs: (batch, max_computing_depth)
        # alive_masks: (batch, max_computing_depth)
        halting_probs = torch.stack(halting_prob_list, dim=-1).float()
        alive_masks = torch.stack(alive_mask_list, dim=-1).float()
        merged_hidden = self._get_merged_state(halting_probs, hidden_list)

        accumulated_halting_probs = (halting_probs * alive_masks).sum(1)
        num_updated = alive_masks.sum(1)

        return merged_hidden, accumulated_halting_probs, num_updated

    def get_output_state(self, hidden):
        if self._rnn_type == RNNType.LSTM:
            return hidden[0]

        if self._rnn_type == RNNType.VanillaRNN or self._rnn_type == RNNType.GRU:
            return hidden

        return None

    def _add_depth_embedding(self, inputs: torch.Tensor, depth: int) -> torch.Tensor:
        if self._depth_embedding_type == DepthEmbeddingType.SINUSOID:
            step_inputs = add_depth_features_to_single_position(inputs, depth)

        elif self._depth_embedding_type == DepthEmbeddingType.LEARNT:
            # every item in batch receives the same depth embedding
            step_inputs = inputs + self._depth_embedding.select(0, depth).unsqueeze(0)

        else: #self._depth_embedding_type == DepthEmbeddingType.NONE:
            step_inputs = inputs

        return step_inputs

    def _get_merged_state(self, halting_probs, hidden_list) -> torch.Tensor:
        """
        Merge the hidden list according to the merging mode.

        :param: halting_probs: (batch, max_computing_depth)
        :param: hidden_list: [(batch, hidden_dim)] or [Tuple[(batch, hidden_dim), (batch, hidden_dim)]]
        :return:
        """
        if self._rnn_type == RNNType.LSTM:
            hidden_vars = zip(*hidden_list)
        else:
            hidden_vars = [hidden_list]

        # only [hidden] or [hidden, context] in LSTM
        hidden = [self._merge_anytime_state(anytime_state, halting_probs) for anytime_state in hidden_vars]

        # simplify returning data for GRU and vanilla RNN
        if len(hidden) == 1:
            hidden = hidden[0]

        return hidden

    def _merge_anytime_state(self, anytime_state, halting_probs):
        # anytime_state: [(batch, hidden_dim)]
        # state: (batch, hidden_dim, max_computing_depth)
        state = torch.stack(anytime_state, dim=2)

        # computation depth is the last dimension
        if self.state_mode == AdaptiveStateMode.BASIC:
            # desired_state: (batch, hidden_dim)
            merged_state = anytime_state[-1]

        elif self.state_mode == AdaptiveStateMode.MEAN_FIELD:
            # halting_probs_extended: (batch, 1, max_computing_depth)
            # desired_state: (batch, hidden_dim)
            halting_probs_extended = halting_probs.unsqueeze(1)
            merged_state = (halting_probs_extended * state).sum(dim=-1)

        elif self.state_mode == AdaptiveStateMode.RANDOM:
            # samples_index: torch.LongTensor: (batch, )
            # desired_state: (batch, hidden_dim)
            samples_index = torch.multinomial(halting_probs, 1).squeeze(-1)
            merged_state = select_item_along_dim(state, samples_index)

        else:
            assert False

        assert len(merged_state.size()) == 2
        assert merged_state.size() == state[:, :, -1].size()

        return merged_state

    def _add_depth_wise_attention(self,
                                  step_inputs: torch.Tensor,
                                  hidden: torch.Tensor,
                                  previous_hidden_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Get depth-wise attention over the previous hiddens

        :param step_inputs: (batch, hidden_dim)
        :param hidden: (batch, hidden_dim)
        :param previous_hidden_list: [(batch, hidden_dim)]
        :return: (batch, hidden_dim)
        """
        if self._depth_wise_attention is None:
            return hidden

        # attend_over: (batch, steps, hidden_dim)
        output_states = [self.get_output_state(hidden) for hidden in previous_hidden_list]
        attend_over = torch.stack(output_states, dim=1)

        # context: (batch, 1, hidden_dim)
        context = self._depth_wise_attention(step_inputs, attend_over)

        step_inputs = step_inputs + context
        step_inputs = step_inputs.squeeze(1)

        return step_inputs

    def init_hidden_states(self, source_state, source_mask, is_bidirectional=False):
        batch, _, hidden_dim = source_state.size()

        last_word_indices = source_mask.sum(1).long() - 1
        expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch, 1, hidden_dim)
        final_encoder_output = source_state.gather(1, expanded_indices)
        final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, hidden_dim)

        if is_bidirectional:
            hidden_dim = hidden_dim // 2
            final_encoder_output = final_encoder_output[:, :hidden_dim]

        initial_hidden = final_encoder_output
        initial_context = torch.zeros_like(initial_hidden)

        if self._rnn_type == RNNType.LSTM:
            return initial_hidden, initial_context
        else:
            return initial_hidden

