import torch
from torch import nn
from .dense import Re2Dense

class Re2Prediction(nn.Module):
    def __init__(self, mode: str, inp_sz: int, hid_sz: int, num_classes: int, dropout: float = .2, activation=nn.ReLU()):
        super().__init__()
        self.mode = mode.lower()
        if mode in ("full", "symmetric"):
            feat_sz = inp_sz * 4
        else:
            feat_sz = inp_sz * 2

        self.dense1 = Re2Dense(feat_sz, hid_sz, activation=activation)
        self.dense2 = Re2Dense(hid_sz, num_classes, activation=None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b):
        feature = self._get_features(a, b)
        feature = self.dropout(feature)
        feature = self.dense1(feature)
        feature = self.dropout(feature)
        prediction = self.dense2(feature)
        return prediction

    def _get_features(self, a, b):
        if self.mode == "full":
            return torch.cat([a, b, a * b, a - b], dim=-1)
        elif self.mode == "symmetric":
            return torch.cat([a, b, a * b, torch.abs(a - b)], dim=-1)
        else:
            return torch.cat([a, b], dim=-1)



