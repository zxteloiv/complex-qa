from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F
from models.modules.stacked_rnn_cell import StackedRNNCell, RNNType

class NTDecoder(nn.Module):
    """Non-Terminal Decoder"""
    def __init__(self, rnn_cell: StackedRNNCell, init_lowest: bool = True, normalize_nt: bool = True):
        """
        :param rnn_cell: a stacked RNN cell instance.
        :param init_lowest: only initialize the hidden vector of the lowest rnn cell, or not.
        :param normalize_nt: to normalize the non-terminal vectors or not
        """
        super().__init__()
        self.decoder = rnn_cell
        self.init_lowest = init_lowest
        self.normalize = F.normalize if normalize_nt else (lambda x: x)

    def forward(self, h_nt, h_t = None):
        """
        :param h_nt: (batch, hidden_dim)
        :param h_t: (batch, hidden_dim)
        :return: (batch, 2, hidden_dim), the codes to quantize later
        """
        num_layer = self.decoder.get_layer_num()
        h_t = torch.zeros_like(h_nt) if h_t is None else h_t

        if self.init_lowest:
            init_state = [h_t] + [torch.zeros_like(h_t) for _ in range(num_layer - 1)]
        else:
            init_state = [h_t for _ in range(num_layer)]

        init_state, _ = self.decoder.init_hidden_states(init_state)

        h_nt = self.normalize(h_nt)
        state, o1 = self.decoder(h_nt, init_state)
        o1 = self.normalize(o1)
        state, o2 = self.decoder(o1, state)
        o2 = self.normalize(o2)

        # (batch, 2, hidden_dim)
        return torch.stack([o1, o2], dim=1)

