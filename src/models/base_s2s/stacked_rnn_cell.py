from typing import List, Optional, Union
import torch
import torch.nn
from models.modules.torch_rnn_wrapper import TorchRNNWrapper as HSWrapper, RNNType
from utils.nn import filter_cat
from models.interfaces.unified_rnn import UnifiedRNN
from ..modules.variational_dropout import VariationalDropout

class StackedRNNCell(torch.nn.Module):
    def __init__(self, rnns: List[UnifiedRNN], dropout: float = 0.):
        super(StackedRNNCell, self).__init__()
        self._input_dropouts = [VariationalDropout(dropout, on_the_fly=False) for _ in range(len(rnns) - 1)]
        self._hidden_dropouts = [VariationalDropout(dropout, on_the_fly=False) for _ in range(len(rnns))]
        for rnn, vd in zip(rnns, self._hidden_dropouts):
            rnn.set_hx_dropout_fn(vd)
        self.layer_rnns = torch.nn.ModuleList(rnns)

    def get_layer_num(self):
        return len(self.layer_rnns)

    def reset(self):
        for m in self._input_dropouts + self._hidden_dropouts:
            m.reset()

    def forward(self, inputs, hidden):
        last_layer_output = inputs
        updated_hiddens = []
        for i, rnn in enumerate(self.layer_rnns):
            if i > 0:
                last_layer_output = self._input_dropouts[i](last_layer_output)

            layer_hidden, last_layer_output = rnn(last_layer_output, None if hidden is None else hidden[i])
            updated_hiddens.append(layer_hidden)

        return updated_hiddens, last_layer_output

    def get_output_state(self, hidden):
        last_hidden = hidden[-1]
        return self.layer_rnns[-1].get_output_state(last_hidden)

    def merge_hidden_list(self, hidden_list, weight):
        # hidden_list: [hidden_1, ..., hidden_T] along timestep
        # hidden: (layer_hidden_1, ..., layer_hidden_L) along layers
        layered_list = zip(*hidden_list)
        merged = [self.layer_rnns[i].merge_hidden_list(layer_hidden, weight)
                  for i, layer_hidden in enumerate(layered_list) ]
        return merged

    def init_hidden_states(self, layer_hidden: List[torch.Tensor]):
        assert len(layer_hidden) == self.get_layer_num()
        hiddens = []
        for rnn, init_hidden in zip(self.layer_rnns, layer_hidden):
            h, _ = rnn.init_hidden_states(init_hidden)
            hiddens.append(h)
        return hiddens, self.get_output_state(hiddens)


class StackedLSTMCell(StackedRNNCell):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout = 0.):
        super().__init__([
            HSWrapper(torch.nn.LSTMCell(input_dim if floor == 0 else hidden_dim, hidden_dim))
            for floor in range(n_layers)
        ], dropout)

class StackedGRUCell(StackedRNNCell):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout = 0.):
        super().__init__([
            HSWrapper(torch.nn.GRUCell(input_dim if floor == 0 else hidden_dim, hidden_dim))
            for floor in range(n_layers)
        ], dropout)

if __name__ == '__main__':
    batch, dim, L = 5, 10, 2
    cell = StackedLSTMCell(dim, L, L)
    f0 = torch.randn(batch, L).float()
    f1 = torch.randn(batch, L).float()
    h, o = cell.init_hidden_states([f0, f1])

    assert o.size() == (batch, L)
    assert len(h) == L

    x = torch.randn(batch, dim)

    hs = []
    T = 5
    for _ in range(T):
        h, o = cell(x, h)
        assert o.size() == (batch, L)
        assert len(h) == L
        hs.append(h)

    weight: torch.Tensor = torch.randn(batch, T)
    weight = torch.nn.Softmax(dim=1)(weight)

    mh = cell.merge_hidden_list(hs, weight)




