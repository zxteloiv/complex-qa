import torch.nn as nn


class ResLayer(nn.Module):
    def __init__(self, in_out_dim, deprecated_dim=None,
                 activation: nn.Module = nn.Mish(),
                 dropout: nn.Module = None,
                 ):
        super(ResLayer, self).__init__()
        if deprecated_dim is not None:
            assert in_out_dim == deprecated_dim, 'ResLayer will not transform dimensions anymore.'

        self.linear = nn.Linear(in_out_dim, in_out_dim)
        self.act = activation
        self.dropout = dropout

    def forward(self, x):
        z = self.linear(x)
        z = self.act(z)
        if self.dropout:
            z = self.dropout(z)
        return x + z

