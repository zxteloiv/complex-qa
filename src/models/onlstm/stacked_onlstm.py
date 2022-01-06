import torch
import torch.nn.functional
from torch import nn
from .onlstm import ONLSTMCell


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dropout=0.5):
        if not self.training or not dropout:
            return x

        mask = x.new_empty((1, x.size()[1], x.size()[2])).bernoulli_(1 - dropout) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class ONLSTMStack(nn.Module):
    def __init__(self, layer_sizes, chunk_size, dropout=0., dropconnect=0.):
        super(ONLSTMStack, self).__init__()
        self.cells = nn.ModuleList([ONLSTMCell(layer_sizes[i],
                                               layer_sizes[i+1],
                                               chunk_size,
                                               dropconnect=dropconnect)
                                    for i in range(len(layer_sizes) - 1)])
        self.lockdrop = LockedDropout()
        self.dropout = dropout
        self.sizes = layer_sizes

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells]

    def forward(self, input, hidden):
        length, batch_size, _ = input.size()

        if self.training:
            for c in self.cells:
                c.sample_masks()

        prev_state = list(hidden)
        prev_layer = input

        raw_outputs = []
        outputs = []
        distances_forget = []
        distances_in = []
        for l in range(len(self.cells)):
            curr_layer = [None] * length
            dist = [None] * length
            t_input = self.cells[l].ih(prev_layer)

            for t in range(length):
                hidden, cell, d = self.cells[l](
                    None, prev_state[l],
                    transformed_input=t_input[t]
                )
                prev_state[l] = hidden, cell  # overwritten every timestep
                curr_layer[t] = hidden
                dist[t] = d

            prev_layer = torch.stack(curr_layer)
            dist_cforget, dist_cin = zip(*dist)
            dist_layer_cforget = torch.stack(dist_cforget)
            dist_layer_cin = torch.stack(dist_cin)
            raw_outputs.append(prev_layer)
            if l < len(self.cells) - 1:
                prev_layer = self.lockdrop(prev_layer, self.dropout)
            outputs.append(prev_layer)
            distances_forget.append(dist_layer_cforget)
            distances_in.append(dist_layer_cin)
        output = prev_layer

        return output, prev_state, raw_outputs, outputs, (torch.stack(distances_forget), torch.stack(distances_in))


if __name__ == "__main__":
    x = torch.Tensor(10, 10, 10)
    x.data.normal_()
    lstm = ONLSTMStack([10, 10, 10], chunk_size=10)
    print(lstm(x, lstm.init_hidden(10))[1])

