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
        logPx, kl, x_tree_hid = self.pcfg.encode(tokens)
        batch, h_dim = x_tree_hid.size()
        state = [x_tree_hid.unsqueeze(1)]   # [(b, 1, hid)]
        state_mask = x_tree_hid.new_ones(batch, 1, dtype=torch.long)

        self._loss = (-logPx + kl).mean()
        return state, state_mask

    def get_loss(self):
        return self._loss

    def is_bidirectional(self):
        return False

    def get_output_dim(self) -> int:
        return self.pcfg.encoding_dim
