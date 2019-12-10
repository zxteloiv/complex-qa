import torch
from torch import nn

class Re2Encoder(nn.Module):
    def __init__(self, stacked_conv, activation = None, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.stacked_conv = stacked_conv
        self.activation = activation

    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        :param x: (batch, len, embedding)
        :param mask: (batch, len)
        :return: (batch, len, embedding)
        """
        inp = x * mask.unsqueeze(-1).float()
        inp = inp.transpose(1, 2)   # convert to (batch, embedding, len) for conv1d: (N, C, L)
        for conv1d in self.stacked_conv:
            inp = conv1d(inp)   # output channel should be the same as the input
            if self.activation:
                inp = self.activation(inp)
            inp = self.dropout(inp)

        rtn = inp.transpose(1, 2)   # back to (batch, len, embedding)
        return rtn

    @staticmethod
    def from_re2_default(num_stacked_layers: int,
                         inp_sz: int,
                         hidden: int,
                         kernel_sz: int,
                         dropout: float,
                         activation=nn.ReLU()):
        """
        :param num_stacked_layers: the encoder is actually stacked conv layers
        :param inp_sz: input size, which is embedding size or augmented residual size
        :param hidden:
        :param kernel_sz:
        :param dropout:
        :return:
        """
        stacked_conv = nn.ModuleList([
            nn.Conv1d(inp_sz if i == 0 else hidden, hidden, kernel_sz, padding=kernel_sz//2)
            for i in range(num_stacked_layers)
        ])
        return Re2Encoder(stacked_conv, activation=activation, dropout=dropout)

