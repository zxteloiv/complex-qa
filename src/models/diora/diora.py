import torch
import torch.nn as nn
from base_model import DioraBase

from net_utils import get_inside_states, inside_fill_chart
from net_utils import get_outside_states, outside_fill_chart
from inside_index import build_inside_component_lookup
from net_utils import BatchInfo


# Composition Functions
class ComposeMLP(nn.Module):
    def __init__(self, size, activation, n_layers=2):
        super(ComposeMLP, self).__init__()

        self.size = size
        self.activation = activation
        self.n_layers = n_layers

        self.W = nn.Parameter(torch.FloatTensor(2 * self.size, self.size))
        self.B = nn.Parameter(torch.FloatTensor(self.size))

        for i in range(1, n_layers):
            setattr(self, 'W_{}'.format(i), nn.Parameter(torch.FloatTensor(self.size, self.size)))
            setattr(self, 'B_{}'.format(i), nn.Parameter(torch.FloatTensor(self.size)))
        self.reset_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        device = self.device
        return device.index is not None and device.index >= 0

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, hs):
        input_h = torch.cat(hs, 1)
        h = torch.matmul(input_h, self.W)
        h = self.activation(h + self.B)
        for i in range(1, self.n_layers):
            W = getattr(self, 'W_{}'.format(i))
            B = getattr(self, 'B_{}'.format(i))
            h = self.activation(torch.matmul(h, W) + B)

        return h


# Score Functions

class Bilinear(nn.Module):
    def __init__(self, size_1, size_2=None):
        super(Bilinear, self).__init__()
        self.size_1 = size_1
        self.size_2 = size_2 or size_1
        self.mat = nn.Parameter(torch.FloatTensor(self.size_1, self.size_2))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, vector1, vector2):
        # bilinear
        # a = 1 (in a more general bilinear function, a is any positive integer)
        # vector1.shape = (b, m)
        # matrix.shape = (m, n)
        # vector2.shape = (b, n)
        bma = torch.matmul(vector1, self.mat).unsqueeze(1)
        ba = torch.matmul(bma, vector2.unsqueeze(2)).view(-1, 1)
        return ba


# Base
class DioraMLP(DioraBase):
    K = 1

    def __init__(self, *args, **kwargs):
        self.n_layers = kwargs.get('n_layers', None)
        super(DioraMLP, self).__init__(*args, **kwargs)

    def init_parameters(self):
        # Model parameters for transformation required at both input and output
        self.inside_score_func = Bilinear(self.size)
        self.outside_score_func = Bilinear(self.size)
        self.root_vector_out_h = nn.Parameter(torch.FloatTensor(self.size))

        self.inside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers)
        self.outside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers)

    def init_with_batch(self, h, info=None):
        super().init_with_batch(h, info)
        self.cache['inside_s_components'] = {i: {} for i in range(self.length)}

    def inside_func(self, batch_info):
        B = batch_info.batch_size
        L = batch_info.length - batch_info.level
        N = batch_info.level
        chart, index = self.chart, self.index

        lh, rh = get_inside_states(batch_info, chart['inside_h'], index, batch_info.size)
        ls, rs = get_inside_states(batch_info, chart['inside_s'], index, 1)

        h = self.inside_compose_func([lh, rh])
        xs = self.inside_score_func(lh, rh)

        s = xs + ls + rs
        s = s.view(B, L, N, 1)
        p = torch.softmax(s, dim=2)

        hbar = torch.sum(h.view(B, L, N, -1) * p, 2)
        hbar = self.inside_normalize_func(hbar)
        sbar = torch.sum(s * p, 2)

        inside_fill_chart(batch_info, chart, index, hbar, sbar)

        self.private_inside_hook(batch_info.level, h, s, p, xs, ls, rs)
        return h, s, p, xs, ls, rs

    def private_inside_hook(self, level, h, s, p, x_s, l_s, r_s):
        """
        This method is meant to be private, and should not be overriden.
        Instead, override `inside_hook`.
        """
        if level == 0:
            return

        length = self.length
        B = self.batch_size
        L = length - level

        x_s = x_s.view(*s.shape)
        assert s.shape == (B, L, level, 1), s.shape
        smax = s.max(dim=2, keepdim=True)[0]
        s = s - smax

        for pos in range(L):
            self.cache['inside_s_components'][level][pos] = s[:, pos, :]

        component_lookup = build_inside_component_lookup(self.index, BatchInfo(length=length, level=level))
        argmax = x_s.argmax(dim=2)
        for i_b in range(B):
            for pos in range(L):
                n_idx = argmax[i_b, pos].item()
                l_level, l_pos, r_level, r_pos = component_lookup[(pos, n_idx)]

                self.cache['inside_tree'][(i_b, 0)][(level, pos)] = \
                    self.cache['inside_tree'][(i_b, 0)][(l_level, l_pos)] + \
                    self.cache['inside_tree'][(i_b, 0)][(r_level, r_pos)] + \
                    [(level, pos)]

    def outside_func(self, batch_info):
        index = self.index
        chart = self.chart

        B = batch_info.batch_size
        L = batch_info.length - batch_info.level

        ph, sh = get_outside_states(
            batch_info, chart['outside_h'], chart['inside_h'], index, batch_info.size)
        ps, ss = get_outside_states(
            batch_info, chart['outside_s'], chart['inside_s'], index, 1)

        h = self.outside_compose_func([sh, ph])
        xs = self.outside_score_func(sh, ph)

        s = xs + ss + ps
        s = s.view(B, -1, L, 1)
        p = torch.softmax(s, dim=1)
        N = s.shape[1]

        hbar = torch.sum(h.view(B, N, L, -1) * p, 1)
        hbar = self.outside_normalize_func(hbar)
        sbar = torch.sum(s * p, 1)

        outside_fill_chart(batch_info, chart, index, hbar, sbar)
        return h, s, p, xs, ps, ss

