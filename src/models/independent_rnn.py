from typing import List, Mapping, Dict, Optional, Tuple, Union, Callable, Sequence
import torch.nn
import torch.nn.modules

class IndRNNCell(torch.nn.Module):
    """
    An independent recurrent unit.

    .. math::
       \begin{array}
       h_t = \sigma(W x_t + u \bigodot h_{t-1} + b)
       \end{array}
    """
    def __init__(self,
                 inputs_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 activation: torch.nn.Module = torch.nn.modules.SELU()):
        super(IndRNNCell, self).__init__()
        self.inputs_size = inputs_size
        self.hidden_size = hidden_size

        self.activation = activation

        self.weight = torch.nn.Parameter(torch.Tensor(hidden_size, inputs_size))
        self.u_vec = torch.nn.Parameter(torch.Tensor(1, hidden_size))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size)) if bias is not None else None

    def forward(self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        :param inputs: (batch, input_size)
        :param hidden: (batch, hidden_size)
        :return:
        """
        # weight: (hidden, input_size) -> (    1,     hidden, input_size)
        # inputs: (batch, input_size) ->  (batch, input_size,          1)
        # Wx: (batch, hidden, 1) -> (batch, hidden)
        Wx = torch.matmul(self.weight.unsqueeze(0), inputs.unsqueeze(-1)).squeeze(-1)

        # uh: (batch, hidden) <- (1, hidden) * (batch, hidden)
        uh = self.u_vec * hidden if hidden else 0

        raw = Wx + uh
        if self.bias is not None:
            raw = raw + self.bias

        return self.activation(raw)

