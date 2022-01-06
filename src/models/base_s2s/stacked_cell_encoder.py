from typing import List, Any, Optional
import torch.nn
from models.base_s2s.stacked_rnn_cell import StackedRNNCell, UnifiedRNN
from utils.seq_collector import SeqCollector
from ..interfaces.unified_rnn import EncoderRNNStack, RNNStack


class StackedCellEncoder(EncoderRNNStack):
    """Encoder for an input sequence, built from the UnifiedRNN cells interfaces, mimicking the StackedEncoder API"""

    def get_layer_num(self) -> int:
        return self.s_cell.get_layer_num()

    def get_input_dim(self) -> int:
        return self.s_cell.get_input_dim()

    def get_output_dim(self) -> int:
        return self.s_cell.get_output_dim()

    def __init__(self, stacked_cell: RNNStack, output_every_layer=True):
        super().__init__()
        self.s_cell = stacked_cell
        self.output_every_layer = output_every_layer

    def forward(self, seq, mask=None, hx: Optional[List[Any]] = None):
        """
        :param seq: (batch, seq_len, input_dim)
        :param mask: (batch, seq_len), not required for non-bidirectional cells
        :param hx:
        :return:
        """
        mem = SeqCollector()
        for step in range(seq.size()[1]):
            step_input = seq[:, step]
            hx, out = self.s_cell(step_input, hx)
            mem(out=out, hx=hx)

        # out: (batch, len, hid)
        enc_output = mem.get_stacked_tensor('out', dim=1)
        if self.output_every_layer:
            # layered_out: a list of (batch, len, hid)
            layered_output = self._get_layered_output_from_hidden_list(mem['hx'])
            return enc_output, layered_output
        else:
            return enc_output

    def _get_layered_output_from_hidden_list(self, hx_timesteps):
        layered_output = []
        output_along_steps = [self.s_cell.get_layered_output_state(hx) for hx in hx_timesteps]
        for layer in range(self.get_layer_num()):
            output = [step_out[layer] for step_out in output_along_steps]
            seq_output = torch.stack(output, dim=1)     # (batch, seq_len, hid)
            layered_output.append(seq_output)
        return layered_output

    def is_bidirectional(self):
        # the bidirectional encoder will require rnn
        return False
