from ..base_s2s.rnn_stacker import RNNCellStacker
from .onlstm import ONLSTMCell, VariationalDropout


class StackedONLSTM(RNNCellStacker):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 chunk_size: int,
                 n_layers: int,
                 dropout: float = 0.):
        super().__init__([
            ONLSTMCell(input_dim if floor == 0 else hidden_dim, hidden_dim, chunk_size,
                       dropout=VariationalDropout(dropout))
            for floor in range(n_layers)
        ], dropout)
