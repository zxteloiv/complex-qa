import torch.nn as nn


class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Mish(),
            nn.Linear(out_dim, out_dim),
            nn.Mish(),
        )

    def forward(self, x):
        return self.linear(x) + x

