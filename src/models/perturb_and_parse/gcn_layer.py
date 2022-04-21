import torch
from torch import nn
from allennlp.nn import Activation


class GCN(nn.Module):
    def __init__(self, inp_sz: int, hid_sz: int, activation: str = 'relu'):
        super().__init__()
        # in the original paper, all these transformations are one-layer MLP
        self.trans = nn.Linear(inp_sz, hid_sz * 3)
        self.inp_sz = inp_sz
        self.hid_sz = hid_sz
        self.act = Activation.by_name(activation)()

    def forward(self, x: torch.Tensor, connectivity: torch.Tensor):
        """
        Forward GCN to update each node representation.
        :param x: (batch, seq, inp_sz)
        :param connectivity: (batch, seq, seq)
        :return:
        """
        # x_*: (batch, seq, hid_sz)
        # except for the xi not to interact with the graph, the xh and xo are symmetric,
        # but below we explain it concretely to make it easier to understand
        xi, xh, xo = self.trans(x).split(self.hid_sz, dim=-1)

        # let the arc direction be xh -> xo
        # xh: (batch, heads, hid_sz), the node representations when acting as arrow heads
        # xo: (batch, tails, hid_sz), the node representations when acting as arrow tails

        # semantically let
        # connectivity: (batch, heads, tails), where heads=tails=seq
        # msg_o: (batch, heads, hid_sz), the message passed from all tails for each head
        # msg_h: (batch, tails, hid_sz), the message passed from all heads for each tail
        msg_o = torch.bmm(connectivity, xo)
        msg_h = torch.bmm(connectivity.transpose(1, 2), xh)

        return self.act(xi + msg_o + msg_h)
