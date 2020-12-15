from typing import Optional, Literal
import torch
from torch import nn
import torch.nn.functional as F

class MoSProjection(nn.Module):
    def __init__(self, mixture_num: int, input_dim, output_dim, flatten_softmax: bool = False,
                 output_semantics: Literal["logits", "probs"] = "logits", eps: float = 1e-15):
        super(MoSProjection, self).__init__()
        assert mixture_num >= 1

        self.mixture_num = mixture_num
        self.input_dim = input_dim
        self.output_dim = output_dim

        self._flatten_softmax: bool = flatten_softmax

        self.mixture_weight_layer = nn.Sequential(
            nn.Linear(input_dim, mixture_num),
            nn.Softmax(dim=-1)
        )

        self.mixture_proj_layer = nn.Linear(input_dim, output_dim * mixture_num)
        self.output_semantics = output_semantics
        self.eps = eps

    def forward(self,
                proj_input: torch.Tensor,
                logit_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # size_prefix: (batch, *)
        size_prefix = proj_input.size()[:-1]
        batch_size = size_prefix[0]

        # bias shape is required to be: either (batch, vocab),
        # or (batch, *, vocab) which is exact the same as proj_input except the last dimension
        assert (logit_bias is None
            or logit_bias.size() == size_prefix + (self.output_dim,)
            or logit_bias.size() == (proj_input.size()[0], self.output_dim)
        )

        # proj_input: (batch, *, input_dim)
        # mixture: (batch, *, output_dim * mixture_num)
        # weight: (batch, *, mixture_num)
        mixture = self.mixture_proj_layer(proj_input)
        weight: torch.Tensor = self.mixture_weight_layer(proj_input)

        # mixture_rs: (batch, *, output_dim, mixture_num)
        mixture_rs = mixture.reshape(size_prefix + (self.output_dim, self.mixture_num))

        if logit_bias is not None:
            # logit_bias: (batch, [*,] output_dim) -> (batch, *, output_dim)
            while logit_bias.ndimension() < proj_input.ndimension():
                logit_bias = logit_bias.unsqueeze(-2)
            # logit_bias: (batch, *, output_dim) -> (batch, *, output_dim, 1)
            logit_bias = logit_bias.unsqueeze(-1)
            mixture_rs = mixture_rs + logit_bias

        if self._flatten_softmax:
            orig_shape = mixture_rs.size()
            mixture_rs = mixture_rs.reshape(batch_size, -1, self.mixture_num).softmax(-2).reshape(orig_shape)
        else:
            mixture_rs = mixture_rs.softmax(-2)

        weight_rs = weight.reshape(size_prefix + (1, self.mixture_num))

        # output: (batch, *, output_dim)
        output = (mixture_rs * weight_rs).sum(-1)
        del mixture_rs, weight_rs
        if self.output_semantics == "logits":
            output = (output + self.eps).log()
        return output
