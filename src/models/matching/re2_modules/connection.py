import torch
from torch import nn
from .dense import Re2Dense
import math

class Re2Conn(nn.Module):
    def __init__(self, mode: str, emb_sz: int, hid_sz: int):
        """
        :param mode: Chosen from "none", "residual", and "aug"
        """
        super().__init__()
        self.mode = mode.lower()
        self.res_proj = None
        if mode == "residual":  # only needed in residual mode
            self.res_proj = Re2Dense(hid_sz, emb_sz)
        self.hid_sz = hid_sz
        self.emb_sz = emb_sz

    def forward(self, x, res, layer_depth: int):
        if self.mode == "none":
            return x
        elif self.mode == "residual":
            return (self.res_proj(x) + res) * math.sqrt(0.5)

        # "aug" mode
        # at layer 0, no residual connection could appear
        # at layer 1, residual connection is the embedding itself
        # for layer >1, residual connection will be added before embedding is concatenated
        assert layer_depth >= 1
        if layer_depth == 1:
            raw_emb = res
        else:
            raw_emb = res[:, :, :-self.hid_sz]
            x = (x + res[:, :, -self.hid_sz:]) * math.sqrt(0.5)

        return torch.cat([raw_emb, x], dim=-1)

    def get_output_size(self):
        if self.mode == "none":
            return self.hid_sz  # same size with the output of fusion
        elif self.mode == "residual":
            return self.emb_sz  # same size with res connection, starting from embedding
        elif self.mode == "aug":
            return self.emb_sz + self.hid_sz

