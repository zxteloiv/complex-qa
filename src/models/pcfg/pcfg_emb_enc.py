from typing import Tuple, List

import torch

from models.interfaces.encoder import EmbedAndEncode
from models.pcfg.C_PCFG import CompoundPCFG


class PCFGEmbedEncode(EmbedAndEncode):

    def __init__(self, pcfg: CompoundPCFG):
        super().__init__()
        self.pcfg = pcfg
        self._loss = None

    def forward(self, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # mem: (b, n * (n - 1), hid)
        logPx, kl, mem = self.pcfg.encode(tokens)
        state = [mem]
        state_mask = self.get_state_mask(tokens)

        self._loss = (-logPx + kl).mean()
        return state, state_mask

    def get_loss(self):
        return self._loss

    def get_state_mask(self, tokens):
        b, n = tokens.size()
        lengths = (tokens != self.pcfg.padding).long().sum(-1)
        length_chart_mask = tokens.new_ones(b, n, n)
        # iterate over batch instead of instantiating a length mask lookup
        for length, chart in zip(lengths, length_chart_mask):
            chart.tril_(length - n)
        length_chart_mask = length_chart_mask.rot90(-1, (1, 2))
        length_chart_mask = length_chart_mask[:, 1:, :].reshape(b, n * (n - 1))
        return length_chart_mask

    def is_bidirectional(self):
        return False

    def get_output_dim(self) -> int:
        return self.pcfg.encoding_dim
