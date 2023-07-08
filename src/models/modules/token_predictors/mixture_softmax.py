import torch
from torch import nn
from models.interfaces.token_predictor import TokenPredictor, PredSemantics


class MoSPredictor(TokenPredictor):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 mixture_num: int,
                 flatten_softmax: bool = False,
                 eps: float = 1e-15):
        super().__init__()
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
        self.eps = eps

    def forward(self, proj_input: torch.Tensor) -> torch.Tensor:
        # size_prefix: (batch, *)
        size_prefix = proj_input.size()[:-1]
        batch_size = size_prefix[0]

        # proj_input: (batch, *, input_dim)
        # mixture: (batch, *, output_dim * mixture_num)
        # weight: (batch, *, mixture_num)
        mixture = self.mixture_proj_layer(proj_input)
        weight: torch.Tensor = self.mixture_weight_layer(proj_input)

        # mixture_rs: (batch, *, output_dim, mixture_num)
        mixture_rs = mixture.reshape(size_prefix + (self.output_dim, self.mixture_num))

        if self._flatten_softmax:
            orig_shape = mixture_rs.size()
            mixture_rs = mixture_rs.reshape(batch_size, -1, self.mixture_num).softmax(-2).reshape(orig_shape)
        else:
            mixture_rs = mixture_rs.softmax(-2)

        weight_rs = weight.reshape(size_prefix + (1, self.mixture_num))

        # output: (batch, *, output_dim)
        output = (mixture_rs * weight_rs).sum(-1)
        del mixture_rs, weight_rs

        if self.output_semantic == PredSemantics.logits:
            output = (output + self.eps).log()
        return output
