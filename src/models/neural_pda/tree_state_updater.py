import torch
from torch import nn
import torch.nn.functional as F
from ..interfaces.unified_rnn import UnifiedRNN
from utils.nn import get_final_encoder_states

class TreeStateUpdater(nn.Module):
    def forward(self, tree_state, seq_states, seq_mask):
        """
        :param tree_state: (batch, hid)
        :param seq_states: (batch, hid, seq)
        :param seq_mask: (batch, seq)
        :return: an updated tree_state: (batch, hid)
        """
        if tree_state is None:
            return self.forward_with_empty_tree(seq_states, seq_mask)

        else:
            return self.forward_with_available_tree(tree_state, seq_states, seq_mask)

    def forward_with_empty_tree(self, seq_states, seq_mask):
        """
        :param seq_states: (batch, hid, seq)
        :param seq_mask: (batch, seq)
        :return: an updated tree_state: (batch, hid)
        """
        raise NotImplementedError

    def forward_with_available_tree(self, tree_state, seq_states, seq_mask):
        """
        :param tree_state: (batch, hid)
        :param seq_states: (batch, hid, seq)
        :param seq_mask: (batch, seq)
        :return: an updated tree_state: (batch, hid)
        """
        raise NotImplementedError

class BoundedAddTSU(TreeStateUpdater):
    def __init__(self, min_val: float = 1., max_val: float = 1.):
        super().__init__()
        self._activation = nn.Hardtanh(min_val=min_val, max_val=max_val)

    def forward_with_empty_tree(self, seq_states, seq_mask):
        return self.forward_with_available_tree(0, seq_states, seq_mask)

    def forward_with_available_tree(self, tree_state, seq_states, seq_mask):
        return self._activation(tree_state + (seq_states * seq_mask.unsqueeze(dim=1)).sum(dim=-1))


class OrthogonalAddTSU(TreeStateUpdater):
    def __init__(self, min_val: float = 1., max_val: float = 1.):
        super().__init__()
        self._activation = nn.Hardtanh(min_val=min_val, max_val=max_val)

    def forward_with_empty_tree(self, seq_states, seq_mask):
        return self.forward_with_available_tree(None, seq_states, seq_mask)

    def forward_with_available_tree(self, tree_state, seq_states, seq_mask):
        h = tree_state
        for step in range(seq_mask.size()[-1]):
            step_state = seq_states[:, :, step]             # step_state: (batch, hid)
            step_mask = seq_mask[:, step].unsqueeze(-1)     # step_mask: (batch, 1)

            if h is not None:
                cos_theta = F.cosine_similarity(h, step_state).unsqueeze(-1)    # cos_theta: (batch, 1)
                h = self._activation(h + (step_state - h * cos_theta) * step_mask)
            else:
                h = step_state

        return h

class SeqRNNTSU(TreeStateUpdater):
    def __init__(self, rnn_cell: UnifiedRNN):
        super().__init__()
        self._rnn= rnn_cell

    def forward_with_empty_tree(self, seq_states, seq_mask):
        return self.forward_with_available_tree(None, seq_states, seq_mask)

    def forward_with_available_tree(self, tree_state, seq_states, seq_mask):
        hx = None if tree_state is None else self._rnn.init_hidden_states(tree_state)
        rnn_out_list = []
        for i in range(seq_states.size()[-1]):
            hx, out = self._rnn(seq_states[:, :, i], hx)
            rnn_out_list.append(out)

        # (batch, seq, hid)
        rnn_out = torch.stack(rnn_out_list, dim=1)
        return get_final_encoder_states(rnn_out, seq_mask)

