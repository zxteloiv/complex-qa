import torch.nn as nn


class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        if in_dim != out_dim:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.linear = None

        self.linear2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Mish(),
            nn.Linear(out_dim, out_dim),
            nn.Mish(),
        )

    def forward(self, x):
        if self.linear:
            x = self.linear(x)
        return self.linear2(x) + x

