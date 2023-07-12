import torch
import torch.nn
from models.modules.torch_rnn_wrapper import TorchRNNWrapper as HSWrapper
from models.interfaces.unified_rnn import UnifiedRNN, RNNStack, T_HIDDEN
from ..modules.variational_dropout import VariationalDropout


class RNNCellStacker(RNNStack):
    def __init__(self, rnns: list[UnifiedRNN], dropout: float = 0.):
        super(RNNCellStacker, self).__init__()
        self._input_dropouts = torch.nn.ModuleList([VariationalDropout(dropout, on_the_fly=False)
                                                    for _ in range(len(rnns) - 1)])
        self.layer_rnns = torch.nn.ModuleList(rnns)

    def get_layer_num(self):
        return len(self.layer_rnns)

    def get_input_dim(self):
        return self.layer_rnns[0].get_input_dim()

    def get_output_dim(self):
        return self.layer_rnns[-1].get_output_dim()

    def reset(self):
        for m in self._input_dropouts:
            m.reset()

        for m in self.layer_rnns.modules():
            if isinstance(m, VariationalDropout):
                m.reset()

    def forward(self, inputs, hidden) -> tuple[list[T_HIDDEN], torch.Tensor]:
        last_layer_output = inputs
        updated_hiddens = []
        for i, rnn in enumerate(self.layer_rnns):
            if i > 0:
                last_layer_output = self._input_dropouts[i - 1](last_layer_output)

            layer_hidden, last_layer_output = rnn(last_layer_output, None if hidden is None else hidden[i])
            updated_hiddens.append(layer_hidden)

        return updated_hiddens, last_layer_output

    def get_output_state(self, hidden):
        last_hidden = hidden[-1]
        return self.layer_rnns[-1].get_output_state(last_hidden)

    def get_layered_output_state(self, hidden: list[T_HIDDEN]) -> list[torch.Tensor]:
        assert len(hidden) == len(self.layer_rnns), "hidden must contains the same number of layers as the RNNStack"
        layered_output = []
        for layer_hx, rnn in zip(hidden, self.layer_rnns):
            rnn: UnifiedRNN
            layered_output.append(rnn.get_output_state(layer_hx))

        return layered_output

    def merge_hidden_list(self, hidden_list, weight):
        # hidden_list: [hidden_1, ..., hidden_T] along timestep
        # hidden: (layer_hidden_1, ..., layer_hidden_L) along layers
        layered_list = zip(*hidden_list)
        merged = [self.layer_rnns[i].merge_hidden_list(layer_hidden, weight)
                  for i, layer_hidden in enumerate(layered_list) ]
        return merged

    def init_hidden_states(self, layer_hidden: list[torch.Tensor]):
        assert len(layer_hidden) == self.get_layer_num()
        hiddens = [rnn.init_hidden_states(init_hidden) for rnn, init_hidden in zip(self.layer_rnns, layer_hidden)]
        return hiddens


class StackedLSTMCell(RNNCellStacker):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float = 0.):
        super().__init__([
            HSWrapper(torch.nn.LSTMCell(input_dim if floor == 0 else hidden_dim, hidden_dim),
                      VariationalDropout(dropout))
            for floor in range(n_layers)
        ], dropout)


class StackedGRUCell(RNNCellStacker):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float = 0.):
        super().__init__([
            HSWrapper(torch.nn.GRUCell(input_dim if floor == 0 else hidden_dim, hidden_dim),
                      VariationalDropout(dropout))
            for floor in range(n_layers)
        ], dropout)


if __name__ == '__main__':
    batch, dim, L = 5, 10, 2
    cell = StackedLSTMCell(dim, L, L)
    f0 = torch.randn(batch, L).float()
    f1 = torch.randn(batch, L).float()
    h = cell.init_hidden_states([f0, f1])
    o = cell.get_output_state(h)

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




