from typing import List, Optional
import torch
import torch.nn
from models.modules.universal_hidden_state_wrapper import UniversalHiddenStateWrapper as HSWrapper, RNNType
from utils.nn import filter_cat

class StackedRNNCell(torch.nn.Module):
    def __init__(self, RNNType, input_dim, hidden_dim, n_layers, intermediate_dropout: float = 0.):
        super(StackedRNNCell, self).__init__()

        assert n_layers >= 1
        self.layer_rnns = torch.nn.ModuleList(
            [HSWrapper(RNNType(input_dim, hidden_dim))] +
            [HSWrapper(RNNType(hidden_dim, hidden_dim)) for _ in range(n_layers - 1)]
        )

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self._input_dropout = torch.nn.Dropout(intermediate_dropout)

    def get_layer_num(self):
        return len(self.layer_rnns)

    def forward(self, inputs, hidden, input_aux:Optional[List] = None):
        # hidden is a list of subhidden
        last_layer_output = inputs

        if input_aux is not None:
            last_layer_output = filter_cat([last_layer_output] + input_aux, dim=1)

        if inputs.size()[1] < self.input_dim and input_aux is None:
            raise ValueError('Dimension not match')

        updated_hiddens = []
        for i, rnn in enumerate(self.layer_rnns):
            if i > 0:
                last_layer_output = self._input_dropout(last_layer_output)

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

    def init_hidden_states(self, layer_hidden: List[torch.Tensor]):
        assert len(layer_hidden) == self.get_layer_num()
        hiddens = []
        for rnn, init_hidden in zip(self.layer_rnns, layer_hidden):
            h, _ = rnn.init_hidden_states(init_hidden)
            hiddens.append(h)
        return hiddens, self.get_output_state(hiddens)


class StackedLSTMCell(StackedRNNCell):
    def __init__(self, input_dim, hidden_dim, n_layers, intermediate_dropout = 0.):
        super().__init__(RNNType.LSTM, input_dim, hidden_dim, n_layers, intermediate_dropout)

class StackedGRUCell(StackedRNNCell):
    def __init__(self, input_dim, hidden_dim, n_layers, intermediate_dropout = 0.):
        super().__init__(RNNType.GRU, input_dim, hidden_dim, n_layers, intermediate_dropout)


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




