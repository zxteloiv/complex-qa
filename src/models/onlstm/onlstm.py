from typing import Tuple, Any

import torch.nn.functional as F
import torch.nn as nn
import torch
from ..interfaces.unified_rnn import UnifiedRNN
from ..modules.variational_dropout import VariationalDropout


def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


class ONLSTMCell(UnifiedRNN):
    def get_output_state(self, hidden) -> torch.Tensor:
        hx, cx, dist_cf, dist_cin = hidden
        return hx

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.hidden_size

    def __init__(self, input_size, hidden_size, chunk_size, dropout: VariationalDropout = None):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        full_size = 4 * hidden_size + self.n_chunk * 2

        self.ih = nn.Linear(input_size, full_size, bias=True)
        self.hh = nn.Linear(hidden_size, full_size, bias=True)
        self.h_dropout = dropout if dropout is not None else (lambda x: x)

    def forward(self, inputs, hidden):
        if hidden is None:
            (hx, cx, _, _), _ = self.init_hidden_states(inputs.new_zeros((inputs.size()[0], self.hidden_size)))
        else:
            hx, cx, _, _ = hidden  # distances is not required for recurrent computations, but only for explanation
        n_chunk, chunk_sz = self.n_chunk, self.chunk_size

        gates = self.ih(inputs) + self.hh(self.h_dropout(hx))
        cingate, cforgetgate = gates[:, :n_chunk * 2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:, n_chunk * 2:].view(-1, n_chunk * 4, chunk_sz).chunk(4, 1)

        cingate = 1. - cumsoftmax(cingate)
        cforgetgate = cumsoftmax(cforgetgate)

        distance_cforget = 1. - cforgetgate.sum(dim=-1) / n_chunk
        distance_cin = cingate.sum(dim=-1) / n_chunk

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cell = torch.tanh(cell)
        outgate = torch.sigmoid(outgate)

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell

        hy = outgate * torch.tanh(cy)
        hidden = hy.view(-1, self.hidden_size), cy, distance_cforget, distance_cin
        return hidden, hidden[0]

    def init_hidden_states(self, forward_out: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        h = forward_out
        c = h.new_zeros(h.size()[0], self.n_chunk, self.chunk_size)
        dist = None     # the dist is resided in the internal of the hidden state, extractable by the get_distances API
        hidden = (h, c, dist, dist)
        return hidden, h

    def get_distances(self, hidden):
        return None if hidden is None else hidden[-2:]
