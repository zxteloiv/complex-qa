import torch
from torch import nn

class Re2Dense(nn.Module):
    def __init__(self, in_size, out_size,
                 activation: nn.Module = None,
                 use_bias: bool = True,
                 use_weight_norm: bool = True,
                 use_vanilla_init: bool = False,
                 ):
        super().__init__()
        dense = nn.Linear(in_size, out_size, bias=use_bias)

        if use_vanilla_init:
            # try to use the same initializers as the RE2 official TF code,
            # according to the original `gain = sqrt(2.) if activation is relu else 1`,
            # nonliearity is set to "linear" except "relu" for relu
            if any(cls_name in str(activation.__class__).lower() for cls_name in ('relu', 'gelu')):
                nonlinear = "relu"
            else:
                nonlinear = "linear"
            nn.init.kaiming_normal_(dense.weight, nonlinearity=nonlinear)
            nn.init.zeros_(dense.bias)

        else:
            # default initalizers in nn.Linear class.
            pass

        self.dense = dense
        if use_weight_norm:
            self.dense = nn.utils.weight_norm(self.dense, 'weight')

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation is None:
            return self.dense(x)
        else:
            return self.activation(self.dense(x))

