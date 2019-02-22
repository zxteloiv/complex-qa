from typing import List, Mapping, Dict, Optional, Tuple, Union, Callable, Sequence
import torch.nn

from models.independent_rnn import IndRNNCell

class RNNType:
    VanillaRNN = torch.nn.RNNCell
    LSTM = torch.nn.LSTMCell
    GRU = torch.nn.GRUCell
    IndRNN = IndRNNCell


class UniversalHiddenStateWrapper(torch.nn.Module):
    def __init__(self,
                 rnn_cell: torch.nn.Module,
                 get_output_fn: Callable = None,
                 merge_hidden_list_fn: Callable = None):
        super(UniversalHiddenStateWrapper, self).__init__()
        self._rnn_cell = rnn_cell

        self._get_output_fn = get_output_fn
        self._merge_hidden_list_fn = merge_hidden_list_fn

    def forward(self, inputs, hidden):
        hidden = self._rnn_cell(inputs, hidden)
        output = self.get_output_state(hidden)
        return hidden, output

    def get_output_state(self, hidden):
        if self._get_output_fn is not None:
            return self._get_output_fn(hidden)

        rnn_type = type(self._rnn_cell)
        if rnn_type in (RNNType.VanillaRNN, RNNType.GRU, IndRNNCell, ):
            return hidden
        elif rnn_type == RNNType.LSTM:
            return hidden[0]
        else:
            raise NotImplementedError

    def merge_hidden_list(self, hidden_list, weight):
        """
        Merge the hidden_list using weighted sum.

        :param hidden_list: [hidden] or [(hidden, context)] or else
        :param weight: (batch, total = len(hidden_list) )
        :return: hidden or (hidden, context)
        """
        if self._merge_hidden_list_fn is not None:
            return self._merge_hidden_list_fn(hidden_list, weight)

        # weight: (batch, total=len(hidden_list) )
        rnn_type = type(self._rnn_cell)
        if rnn_type in (RNNType.VanillaRNN, RNNType.GRU, IndRNNCell, ):
            return self.weighted_sum_single_var(hidden_list, weight)

        elif rnn_type == RNNType.LSTM:
            # hidden_list: [(hidden, context)]
            h_list, c_list = zip(*hidden_list)
            merged_h = self.weighted_sum_single_var(h_list, weight)
            merged_c = self.weighted_sum_single_var(c_list, weight)
            return merged_h, merged_c

        else:
            raise NotImplementedError

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

        # returns (hidden, output) or ((hidden, context), output)
        if type(self._rnn_cell) == RNNType.LSTM:
            return (initial_hidden, initial_context), initial_hidden
        else:
            return initial_hidden, initial_hidden

    @staticmethod
    def weighted_sum_single_var(var_list, weight):
        # var_list: [var]
        # var: (batch, hidden_emb)
        # stacked: (batch, hidden_emb, total)
        stacked = torch.stack(var_list, dim=-1)

        # weight: (batch, 1, total) <- (batch, total)
        weight = weight.unsqueeze(1)

        merged = (stacked * weight).sum(2)
        return merged


