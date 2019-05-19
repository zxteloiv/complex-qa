import torch
from torch import nn
import torch.nn.functional as F

class MoSProjection(nn.Module):
    def __init__(self, mixture_num: int, input_dim, output_dim):
        super(MoSProjection, self).__init__()
        assert mixture_num >= 1

        self.mixture_num = mixture_num
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mixture_weight_layer = nn.Sequential(
            nn.Linear(input_dim, mixture_num),
            nn.Softmax(dim=-1)
        )

        self.mixture_proj_layer = nn.Linear(input_dim, output_dim * mixture_num)

    def forward(self, proj_input):
        # proj_input: (batch, *, input_dim)
        # mixture: (batch, *, output_dim * mixture_num)
        # weight: (batch, *, mixture_num)
        mixture = self.mixture_proj_layer(proj_input)
        weight: torch.Tensor = self.mixture_weight_layer(proj_input)
        size_prefix = mixture.size()[:-1]

        mixture_rs = mixture.reshape(size_prefix + (self.mixture_num, self.output_dim)).softmax(-1)
        weight_rs = weight.reshape(size_prefix + (self.mixture_num, 1))

        # output: (batch, *, output_dim)
        output = (mixture_rs * weight_rs).sum(-2)
        del mixture_rs, weight_rs
        return output
