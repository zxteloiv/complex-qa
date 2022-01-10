from typing import List, Any, Optional
import torch.nn
from models.base_s2s.stacked_rnn_cell import StackedRNNCell, UnifiedRNN
from utils.seq_collector import SeqCollector
from ..interfaces.unified_rnn import EncoderRNNStack, RNNStack


class StackedCellEncoder(EncoderRNNStack):
    """Encoder for an input sequence, built from the UnifiedRNN cells interfaces, mimicking the StackedEncoder API"""

    def __init__(self, stacked_cell: RNNStack,
                 output_every_layer: bool = True,
                 backward_stacked_cell: Optional[RNNStack] = None):
        super().__init__()
        self.s_cell = stacked_cell
        self.output_every_layer = output_every_layer
        self.back_s_cell = backward_stacked_cell

    def get_layer_num(self) -> int:
        return self.s_cell.get_layer_num()

    def get_input_dim(self) -> int:
        return self.s_cell.get_input_dim()

    def get_output_dim(self) -> int:
        return self.s_cell.get_output_dim() * 2 if self.is_bidirectional() else self.s_cell.get_output_dim()

    def forward(self, seq, mask=None, hx: Optional[List[Any]] = None):
        """
        :param seq: (batch, seq_len, input_dim)
        :param mask: (batch, seq_len), not required for non-bidirectional cells
        :param hx:
        :return:
        """
        forward_out = self._forward_as_step_seq(seq, list(range(seq.size()[1])), hx, is_reversed=False)
        if not self.is_bidirectional():
            return forward_out

        # when running backwards, we make two assumptions
        # 1) the init hx is the same as the forward pass, because the hx is supposed to enrich the sequence embedding
        # 2) the mask will not get processed explicitly due to the different behavior within the batch, in practice,
        #    the hx will be always 0 if the padding idx is set properly in the nn.Embedding objects
        back_out = self._forward_as_step_seq(seq, list(range(seq.size()[1])), hx, is_reversed=True)
        if self.output_every_layer:
            # enc_output: (batch, len, hid)
            # layered_out: a list of (batch, len, hid)
            enc_output, layered_output = forward_out
            back_enc, back_layered = back_out

            # bi_out: (batch, len, hid * 2)
            # bi_layered_out: a list of (batch, len, hid * 2)
            bi_out = torch.cat([enc_output, back_enc], dim=-1)
            bi_layered_out = [torch.cat([lfo, lbo], dim=-1) for lfo, lbo in zip(layered_output, back_layered)]
            return bi_out, bi_layered_out
        else:
            # forward_out, back_out: (batch, len, hid)
            bi_out = torch.cat([forward_out, back_out], dim=-1)
            return bi_out

    def _forward_as_step_seq(self, seq, steps: List[int], hx: Optional[List[Any]] = None, is_reversed: bool = False):
        cell = self.back_s_cell if is_reversed else self.s_cell
        mem = SeqCollector()
        for step in steps:
            step_input = seq[:, step]
            # hx: [Any]
            # out: (batch, hid)
            hx, out = cell(step_input, hx)
            mem(out=out, hx=hx)

        _proper_reverse = lambda x: list(reversed(x)) if is_reversed else x

        # enc_output: (batch, len, hid)
        enc_output = torch.stack(_proper_reverse(mem['out']), dim=1)
        if not self.output_every_layer:
            return enc_output

        # layered_out: a list of (batch, len, hid)
        layered_output = self._get_layered_output_from_hidden_list(_proper_reverse(mem['hx']))
        return enc_output, layered_output

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
        return self.back_s_cell is not None
