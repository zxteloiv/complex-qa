from typing import Dict, List, Tuple
import torch
from torch import nn
import torch.nn.functional
from ..interfaces.unified_rnn import EncoderRNNStack
from .base_model import DioraBase
from utils.preprocessing import nested_list_numbers_to_tensors

T_BATCH = int
TOP_K = int
T_1stKEY = Tuple[T_BATCH, TOP_K]
LEVEL = int
POS = int
CELL = Tuple[LEVEL, POS]
T_2ndKEY = CELL
T_VAL = List[CELL]
CHART = Dict[T_2ndKEY, T_VAL]

T_INSIDE_TREE = Dict[T_1stKEY, CHART]


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
        self.last_output = self.output_mask = None
        # if the diora output is not the concat, the outside pass must be only enabled during training.
        chart = self.diora(inputs, info={'outside': self.use_concat or self.training})

        if not self.use_concat:
            output, output_mask = self._get_inside_tree_output(chart, mask)
        else:
            output, output_mask = self._get_concat_output(chart, mask)

        self.last_output = output
        self.output_mask = output_mask
        return output

    def _get_inside_tree_output(self, chart, mask):
        cell_idx, cell_mask = self.get_cell_idx(mask)
        # inside_h: (batch, chart_size, hidden)
        inside_h: torch.Tensor = chart['inside_h']
        batch_range = torch.arange(cell_idx.size()[0], device=cell_idx.device).unsqueeze(-1)
        return inside_h[batch_range, cell_idx], cell_mask

    def get_cell_idx(self, mask):
        trees: T_INSIDE_TREE = self.diora.cache['inside_tree']
        offset_cache: Dict[int, int] = self.diora.index.get_offset(mask.size()[1])
        id_lst = []
        lengths: List[int] = mask.sum(-1).tolist()
        for i, l in enumerate(lengths):
            indices: List[int] = []     # raw token embeddings must not be attended
            chart: CHART = trees[i, 0]
            tree = chart[l - 1, 0]
            for cell in tree:
                level, pos = cell
                idx = offset_cache[level] + pos
                indices.append(idx)     # choose the selected cells then
            id_lst.append(indices)

        cell_idx = nested_list_numbers_to_tensors(id_lst, -1, example=mask)
        cell_mask = (cell_idx >= 0).long()
        return cell_idx, cell_mask

    def _get_concat_output(self, chart, mask):
        # *_h: (batch, chart_size, hidden)
        inside_h: torch.Tensor = chart['inside_h']
        outside_h: torch.Tensor = chart['outside_h']

        length = mask.size()[1]

        # output: (batch, length, hidden * 2)
        output = torch.cat([inside_h[:, :length], outside_h[:, :length]], dim=-1)
        return output, mask

    def get_root_cell_idx(self, mask):
        batch_sz, max_length = mask.size()
        offset_cache: Dict[int, int] = self.diora.index.get_offset(max_length)
        # valid_lengths: (batch,)
        # idx: (batch,)
        valid_lengths = mask.sum(-1) - 1
        idx = torch.tensor([offset_cache[lvl] for lvl in valid_lengths.tolist()],
                           dtype=torch.long, device=mask.device)
        return idx

    def is_bidirectional(self) -> bool:
        return False

    def get_input_dim(self) -> int:
        return self.diora.size  # for now the input size is the same as the hidden size

    def get_output_dim(self) -> int:
        return self.diora.size * 2 if self.use_concat else self.diora.size

    def get_layer_num(self) -> int:
        return 1

    def get_last_layered_output(self) -> List[torch.Tensor]:
        return [self.last_output]

    def __init__(self, diora: 'DioraBase', use_concat_memory: bool = False):
        """
        Wrap a Diora instance for the BaseSeq2Seq framework.
        :param diora: an instance of either the Diora or the DioraTopk class
        :param use_concat_memory: if it's set True, the output is the concatenation of the
                inside and outside vectors for each token of a sentence. The output length
                will be equal to the sentence length N then.
                Otherwise, if its value is False, the output is the inside vectors from the
                inside pass of every nodes of the hard decoded tree, based on inside scores.
                The output length will be equal to the N - 1, because the composition tree
                is kept binary.
        """
        super().__init__()
        self.diora = diora
        self.last_output = None
        self.output_mask = None
        self.use_concat = use_concat_memory
