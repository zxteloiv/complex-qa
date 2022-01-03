from typing import List, Optional, Callable, Tuple, Any
import torch.nn

from models.modules.independent_rnn import IndRNNCell
from ..interfaces.unified_rnn import UnifiedRNN
from enum import Enum


class RNNType(Enum):
    VanillaRNN = torch.nn.RNNCell
    LSTM = torch.nn.LSTMCell
    GRU = torch.nn.GRUCell
    IndRNN = IndRNNCell


class TorchRNNWrapper(UnifiedRNN):
    def __init__(self,
                 rnn_cell: torch.nn.Module,
                 get_output_fn: Callable = None,
                 merge_hidden_list_fn: Callable = None,
                 hx_dropout_fn: Callable = None,
                 ):
        super().__init__()
        self._rnn_cell = rnn_cell

        self._get_output_fn = get_output_fn
        self._merge_hidden_list_fn = merge_hidden_list_fn
        self.hx_dropout_fn = hx_dropout_fn

    def forward(self, inputs, hidden):
        hidden = self._dropout_hx(hidden)
        hidden = self._rnn_cell(inputs, hidden)
        output = self.get_output_state(hidden)
        return hidden, output

    def set_hx_dropout_fn(self, hx_dropout_fn):
        self.hx_dropout_fn = hx_dropout_fn

    def get_output_state(self, hidden):
        if self._get_output_fn is not None:
            return self._get_output_fn(hidden)

        rnn_type = type(self._rnn_cell)
        if rnn_type in (RNNType.VanillaRNN.value, RNNType.GRU.value, IndRNNCell, ):
            return hidden
        elif rnn_type == RNNType.LSTM.value:
            return hidden[0]
        else:
            raise NotImplementedError

    def _dropout_hx(self, hx):
        if self.hx_dropout_fn is None:
            return hx

        if type(self._rnn_cell) == RNNType.LSTM.value:
            hx = self.hx_dropout_fn(hx[0]), hx[1]
        else:
            hx = self.hx_dropout_fn(hx)
        return hx

    def merge_hidden_list(self, hidden_list, weight):
        """
        Merge the hidden_list using weighted sum.

        :param hidden_list: [hidden] or [(hidden, context)] or else
        :param weight: (batch, total = len(hidden_list) )
        :return: hidden or (hidden, context), or something else if you know about its internals
        """
        if self._merge_hidden_list_fn is not None:
            return self._merge_hidden_list_fn(hidden_list, weight)

        # weight: (batch, total=len(hidden_list) )
        rnn_type = type(self._rnn_cell)
        if rnn_type in (RNNType.VanillaRNN.value, RNNType.GRU.value, IndRNNCell, ):
            return self.weighted_sum_single_var(hidden_list, weight)

        elif rnn_type == RNNType.LSTM.value:
            # hidden_list: [(hidden, context)]
            h_list, c_list = zip(*hidden_list)
            merged_h = self.weighted_sum_single_var(h_list, weight)
            merged_c = self.weighted_sum_single_var(c_list, weight)
            return merged_h, merged_c

        else:
            raise NotImplementedError

    def init_hidden_states(self, forward_out):
        initial_hidden = forward_out
        initial_context = torch.zeros_like(initial_hidden)

        # returns (hidden, output) or ((hidden, context), output)
        if type(self._rnn_cell) == RNNType.LSTM.value:
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
