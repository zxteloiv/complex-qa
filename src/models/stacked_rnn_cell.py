from typing import List, Mapping, Dict, Optional, Tuple, Union, Callable, Sequence
import torch
import torch.nn
from models.universal_hidden_state_wrapper import UniversalHiddenStateWrapper, RNNType

class StackedRNNCell(torch.nn.Module):
    def __init__(self, RNNType, hidden_dim, n_layers):
        super(StackedRNNCell, self).__init__()

        self.layer_rnns = torch.nn.ModuleList([
            UniversalHiddenStateWrapper(RNNType(hidden_dim, hidden_dim)) for _ in range(n_layers)
        ])

        self.hidden_dim = hidden_dim

    def forward(self, inputs, hidden):
        # hidden is a list of subhidden
        last_layer_output = inputs

        updated_hiddens = []
        for i, rnn in enumerate(self.layer_rnns):
            layer_hidden, last_layer_output = rnn(last_layer_output, hidden[i])
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

    def init_hidden_states(self, forward_out, backward_out):
        init_hidden = []
        for i in range(len(self.layer_rnns)):
            h, _ = self.layer_rnns[i].init_hidden_states(forward_out, backward_out)
            init_hidden.append(h)
        return init_hidden, self.get_output_state(init_hidden)

class StackedLSTMCell(StackedRNNCell):
    def __init__(self, hidden_dim, n_layers):
        super(StackedLSTMCell, self).__init__(RNNType.LSTM, hidden_dim, n_layers)

class StackedGRUCell(StackedRNNCell):
    def __init__(self, hidden_dim, n_layers):
        super(StackedGRUCell, self).__init__(RNNType.GRU, hidden_dim, n_layers)


if __name__ == '__main__':
    batch, dim, L = 5, 10, 2
    cell = StackedLSTMCell(dim, L)
    f_out = torch.randn(batch, dim).float()
    h, o = cell.init_hidden_states(f_out, None)

    assert o.size() == (batch, dim)
    assert len(h) == L

    x = torch.randn(batch, dim)

    hs = []
    T = 5
    for _ in range(T):
        h, o = cell(x, h)
        assert o.size() == (batch, dim)
        assert len(h) == L
        hs.append(h)

    weight: torch.Tensor = torch.randn(batch, T)
    weight = torch.nn.Softmax(dim=1)(weight)

    mh = cell.merge_hidden_list(hs, weight)




