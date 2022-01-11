import torch
from typing import Optional, List, Any
from ..interfaces.unified_rnn import UnifiedRNN, EncoderRNN
from utils.seq_collector import SeqCollector


class CellEncoder(EncoderRNN):
    def forward(self, inputs, mask, hidden) -> torch.Tensor:
        """
        :param seq: (batch, seq_len, input_dim)
        :param mask: (batch, seq_len), not required for non-bidirectional cells
        :param hx: Any
        :return: (batch, seq_len, hid [*2])
        """
        forward_out = self._forward_seq(inputs, hidden)
        if not self.is_bidirectional():
            return forward_out

        # when running backwards, we make two assumptions
        # 1) the init hx is the same as the forward pass, because the hx is supposed to enrich the sequence embedding
        # 2) the mask will not get processed explicitly due to the different behavior within the batch, in practice,
        #    the hx will be always 0 if the padding idx is set properly in the nn.Embedding objects
        back_out = self._forward_seq(inputs, hidden, is_reversed=True)
        # forward_out, back_out: (batch, len, hid)
        # bi_out: (batch, len, hid * 2)
        bi_out = torch.cat([forward_out, back_out], dim=-1)
        return bi_out

    def _forward_seq(self, seq: torch.Tensor, hx: Any = None, is_reversed: bool = False):
        if is_reversed:
            cell = self.back_cell
            steps = list(reversed(range(seq.size()[1])))
        else:
            cell = self.cell
            steps = list(range(seq.size()[1]))

        mem = SeqCollector()
        for step in steps:
            step_input = seq[:, step]
            hx, out = cell(step_input, hx)  # out: (batch, hid)
            mem(out=out)

        def _proper_reverse(x): return list(reversed(x)) if is_reversed else x

        # enc_output: (batch, len, hid)
        return torch.stack(_proper_reverse(mem['out']), dim=1)

    def is_bidirectional(self) -> bool:
        return self.back_cell is not None

    def get_input_dim(self) -> int:
        return self.cell.get_input_dim()

    def get_output_dim(self) -> int:
        return self.cell.get_output_dim() * 2 if self.is_bidirectional() else self.cell.get_output_dim()

    def __init__(self, cell: UnifiedRNN, back_cell: Optional[UnifiedRNN] = None):
        super().__init__()
        self.cell = cell
        self.back_cell = back_cell