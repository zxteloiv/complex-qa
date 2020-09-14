from typing import List, Iterable, Optional
import torch
from torch import nn
from .dense import Re2Dense

class Re2Prediction(nn.Module):
    def __init__(self,
                 mode: str,
                 inp_sz: int,
                 hid_sz: int,
                 num_classes: int,
                 aux_feature_sz: int = 0,
                 dropout: float = .2,
                 activation=nn.ReLU(),
                 ):
        super().__init__()
        self.mode = mode.lower()
        if mode == "full":
            feat_sz = inp_sz * 5
        elif mode == "symmetric":
            feat_sz = inp_sz * 4
        else:
            feat_sz = inp_sz * 2

        feat_sz += aux_feature_sz

        self.dense1 = Re2Dense(feat_sz, hid_sz, activation=activation)
        self.dense2 = Re2Dense(hid_sz, num_classes, activation=None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b, *aux_features: Optional[List[torch.Tensor]], return_repr: bool = False):
        """
        :param a:
        :param b:
        :param aux_features: a list of extra features, all with (batch, feat_dim) structure
        :param return_repr: if true, return the last hidden vector before performing prediction
        :return:
        """
        feature = self._get_features(a, b)

        if aux_features is not None and len(aux_features) > 0:
            feature = torch.cat([feature] + list(aux_features), dim=-1)

        feature = self.dropout(feature)
        feature = self.dense1(feature)
        feature = self.dropout(feature)
        prediction = self.dense2(feature)
        if return_repr:
            return prediction, feature
        else:
            return prediction

    def _get_features(self, a, b) -> torch.Tensor:
        if self.mode == "full":
            return torch.cat([a, b, a * b, a - b, b - a], dim=-1)
        elif self.mode == "symmetric":
            return torch.cat([a, b, a * b, torch.abs(a - b)], dim=-1)
        else:
            return torch.cat([a, b], dim=-1)



