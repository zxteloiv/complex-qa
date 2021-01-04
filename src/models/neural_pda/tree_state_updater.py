import torch
from torch import nn
import torch.nn.functional as F

class TreeStateUpdater(nn.Module):
    def forward(self, tree_state, symbol_state):
        if tree_state is None:
            return symbol_state

        else:
            return self.update_tree_state(tree_state, symbol_state)

    def update_tree_state(self, tree_state, symbol_state):
        raise NotImplementedError

class BoundedAddTSU(TreeStateUpdater):
    def update_tree_state(self, tree_state, symbol_state):
        return F.hardtanh(tree_state + symbol_state)

class OrthogonalAddTSU(TreeStateUpdater):
    def update_tree_state(self, tree_state, symbol_state):
        cos_theta = F.cosine_similarity(tree_state, symbol_state).unsqueeze(-1)
        new_ts = F.hardtanh(tree_state + symbol_state * (1 - cos_theta))
        return new_ts


