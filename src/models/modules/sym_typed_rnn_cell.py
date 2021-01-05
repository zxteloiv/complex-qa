from typing import Tuple, Any, Optional, List

import torch
from torch import nn
import math

from ..interfaces.unified_rnn import UnifiedRNN

class SymTypedRNNCell(UnifiedRNN):
    def _forward_internal(self, inputs, hidden: Optional[torch.Tensor]) -> Tuple[Any, torch.Tensor]:
        """
        :param inputs: (batch, *, in_dim)
        :param hidden: (batch, out_dim)
        :return:
        """
        out = self._input_mapping(inputs)   # (batch, *, out_dim)
        if hidden is not None:
            # h_weight: (out_dim, out_dim)
            h_weight = torch.matmul(self._sym_h_weight.t(), self._sym_h_weight)
            # projected_h: (batch, out_dim)
            projected_h = torch.matmul(hidden, h_weight)

            while projected_h.ndim < out.ndim:
                projected_h = projected_h.unsqueeze(-2)

            # out: (batch, *, out_dim)
            out = out + projected_h

        out = self._activation(out)
        return out, out

    def get_output_state(self, hidden) -> torch.Tensor:
        return hidden

    def init_hidden_states(self, forward_out) -> Tuple[Any, torch.Tensor]:
        return forward_out, forward_out

    def __init__(self, input_dim, output_dim, nonlinearity: str = "tanh"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self._input_mapping = nn.Linear(input_dim, output_dim)

        self._sym_h_weight = nn.Parameter(torch.empty(output_dim, output_dim, dtype=torch.float))

        if nonlinearity.lower() not in ('tanh', 'sigmoid', 'linear'):
            nonlinearity = 'leaky_relu'

        nn.init.kaiming_uniform_(self._sym_h_weight, a=math.sqrt(5), nonlinearity=nonlinearity)

        from allennlp.nn.activations import Activation
        self._activation = Activation.by_name(nonlinearity)()
