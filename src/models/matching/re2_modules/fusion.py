import torch
from torch import nn
from .dense import Re2Dense

class Re2Fusion(nn.Module):
    def __init__(self,
                 inp_sz: int,
                 hid_sz: int,
                 use_full_mode: bool = True,
                 dropout: float = 0.2):
        """
        :param inp_sz: input size of x, and alignment
        :param hid_sz: the output size of fusion
        :param use_full_mode:
        :param dropout:
        """
        super().__init__()
        self.use_full_mode = use_full_mode
        self.feat_mappings = nn.ModuleList([
            Re2Dense(inp_sz * 2, hid_sz, nn.ReLU())
            for _ in range(3)   # 3 heuristic features in total
        ])
        self.projection = Re2Dense(hid_sz * 3, hid_sz, nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, align):
        if self.use_full_mode:
            features = [
                torch.cat([x, align], dim=-1),
                torch.cat([x, x - align], dim=-1),
                torch.cat([x, x * align], dim=-1)
            ]
            mapped_features = [f(x) for f, x in zip(self.feat_mappings, features)]
            concated_features = self.dropout(torch.cat(mapped_features, dim=-1))
            rtn = self.projection(concated_features)

        else:
            features = torch.cat([x, align], dim=-1)
            rtn = self.feat_mappings[0](features)

        return rtn

class NeoFusion(nn.Module):
    def __init__(self, inp_sz: int, hid_sz: int, use_full_mode: bool, dropout: float = .2):
        """
        :param inp_sz:
        :param hid_sz:
        :param use_full_mode:
        :param dropout:
        """
        super().__init__()
        self.full_mode = use_full_mode
        self.feature_mapping = Re2Dense(inp_sz, hid_sz, nn.SELU(), use_bias=True, use_weight_norm=True)
        self.dropout = nn.Dropout(dropout)
        proj_inp = hid_sz * 5 if use_full_mode else hid_sz * 2
        self.proj = Re2Dense(proj_inp, hid_sz, nn.SELU(), use_bias=True, use_weight_norm=True)

    def forward(self, x, align):
        x = self.feature_mapping(x)
        align = self.feature_mapping(align)
        features = [x, align, x - align, align - x, x * align] if self.full_mode else [x, align]
        features = torch.cat(features, dim=-1)
        return self.proj(features)

