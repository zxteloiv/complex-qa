import torch.nn

from ..interfaces.unified_rnn import UnifiedRNN, HiddenAwareRNN, T_HIDDEN
from .variational_dropout import VariationalDropout


class _RNNType:
    VanillaRNN = torch.nn.RNNCell
    LSTM = torch.nn.LSTMCell
    GRU = torch.nn.GRUCell


class TorchRNNWrapper(UnifiedRNN, HiddenAwareRNN):
    def __init__(self,
                 rnn_cell: torch.nn.RNNCellBase,
                 hx_dropout_fn: VariationalDropout = None,
                 ):
        super().__init__()
        self._rnn_cell = rnn_cell
        self.hx_dropout_fn = hx_dropout_fn

    def get_input_dim(self):
        return self._rnn_cell.input_size

    def get_output_dim(self):
        return self._rnn_cell.hidden_size

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None):
        hidden = self._dropout_hx(hidden)
        hidden = self._rnn_cell(inputs, hidden)
        output = self.get_output_state(hidden)
        return hidden, output

    def get_output_state(self, hidden: T_HIDDEN):
        rnn_type = type(self._rnn_cell)
        if rnn_type in (_RNNType.VanillaRNN, _RNNType.GRU, ):
            return hidden
        elif rnn_type == _RNNType.LSTM:
            return hidden[0]
        else:
            raise NotImplementedError

    def _dropout_hx(self, hx):
        if self.hx_dropout_fn is None or hx is None:
            return hx

        if type(self._rnn_cell) == _RNNType.LSTM:
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
        # weight: (batch, total=len(hidden_list) )
        rnn_type = type(self._rnn_cell)
        if rnn_type in (_RNNType.VanillaRNN, _RNNType.GRU, ):
            return self.weighted_sum_single_var(hidden_list, weight)

        elif rnn_type == _RNNType.LSTM:
            # hidden_list: [(hidden, context)]
            h_list, c_list = zip(*hidden_list)
            merged_h = self.weighted_sum_single_var(h_list, weight)
            merged_c = self.weighted_sum_single_var(c_list, weight)
            return merged_h, merged_c

        else:
            raise NotImplementedError

    def init_hidden_states(self, forward_out):
        initial_hidden = forward_out

        # returns (hidden, output) or ((hidden, context), output)
        if type(self._rnn_cell) == _RNNType.LSTM:
            initial_context = torch.zeros_like(initial_hidden)
            return initial_hidden, initial_context
        else:
            return initial_hidden
