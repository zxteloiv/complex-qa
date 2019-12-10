from torch import nn

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

