from typing import Dict, List
import torch
from torch import nn
import torch.nn.functional
from ..interfaces.unified_rnn import EncoderRNNStack
from .base_model import DioraBase


class DioraEncoder(EncoderRNNStack):
    """Faked Stack, actually only one layer of diora is allowed"""

    def forward(self, inputs, mask) -> torch.Tensor:
        """
        run forward with diora, return the
        :param inputs: (batch, length, hid)
        :param mask: (batch, length)
        :param hidden:
        :return:
        """
        self.last_output = None
        chart = self.diora(inputs, info={'outside': self.training})
        root_h = self._get_root_h(chart, mask)
        self.last_output = root_h
        return root_h

    def get_root_cell_idx(self, mask):
        batch_sz, max_length = mask.size()
        offset_cache: Dict[int, int] = self.diora.index.get_offset(max_length)
        # valid_lengths: (batch,)
        # idx: (batch,)
        valid_lengths = mask.sum(-1) - 1
        idx = torch.tensor([offset_cache[lvl] for lvl in valid_lengths.tolist()],
                           dtype=torch.long, device=mask.device)
        return idx

    def _get_root_h(self, chart, mask):
        batch_sz, max_length = mask.size()
        # idx: (batch,)
        idx = self.get_root_cell_idx(mask)
        # root_h: (batch, hid)
        root_h = chart['inside_h'][torch.arange(batch_sz, device=mask.device), idx]
        return root_h.unsqueeze(1)  # (batch, 1, hid)

    def is_bidirectional(self) -> bool:
        return False

    def get_input_dim(self) -> int:
        return self.diora.size  # for now the input size is the same as the hidden size

    def get_output_dim(self) -> int:
        return self.diora.size

    def get_layer_num(self) -> int:
        return 1

    def get_last_layered_output(self) -> List[torch.Tensor]:
        return [self.last_output]

    def __init__(self, diora: 'DioraBase'):
        super().__init__()
        self.diora = diora
        self.last_output = None
