import torch
from torch import nn
from utils.nn import seq_masked_std_mean, seq_masked_var_mean

class Re2Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        """
        :param x: (batch, len, embedding)
        :param mask: (batch, len)
        :return: (batch, embedding)
        """
        return ((mask.unsqueeze(-1) + 1e-45).log() + x).max(dim=1)[0]


class NeoRe2Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        """
        :param x: (batch, len, embedding)
        :param mask: (batch, len)
        :return: (batch, embedding * 3)
        """
        maxpooling = ((mask.unsqueeze(-1) + 1e-45).log() + x).max(dim=1)[0]
        stdpooling, meanpooling = seq_masked_std_mean(x, mask, dim=1)

        return torch.cat([maxpooling, stdpooling, meanpooling], dim=-1)


