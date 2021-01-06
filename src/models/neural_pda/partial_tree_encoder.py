import torch
from torch import nn
from ..interfaces.unified_rnn import UnifiedRNN
from utils.nn import get_final_encoder_states

class StackNodeComposer(nn.Module):
    def forward(self, symbol_emb, symbol_mask):
        """
        :param symbol_emb: (batch, max_cur, emb_sz)
        :param symbol_mask: (batch, max_cur)
        :return:
        """
        raise NotImplementedError

class UnifiedRNNNodeComposer(StackNodeComposer):
    def __init__(self, rnn_cell: UnifiedRNN):
        super().__init__()
        self._rnn = rnn_cell

    def forward(self, symbol_emb, symbol_mask):
        hx = None
        output_list = []
        for step in range(symbol_mask.size()[-1]):
            # out: (batch, hid)
            hx, out = self._rnn(symbol_emb[:, step], hx)
            output_list.append(out)

        output = torch.stack(output_list, dim=1)    # (batch, max_cur, hid)
        # (batch, hid)
        return get_final_encoder_states(output, symbol_mask)
