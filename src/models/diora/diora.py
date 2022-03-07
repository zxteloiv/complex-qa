import torch
import torch.nn.functional
from .base_model import DioraBase

from .net_utils import get_inside_states, inside_fill_chart
from .net_utils import get_outside_states, outside_fill_chart
from .inside_index import build_inside_component_lookup
from .net_utils import BatchInfo


# Base
class Diora(DioraBase):
    def __init__(self, *args, **kwargs):
        super(Diora, self).__init__(*args, **kwargs)

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
        # p = torch.nn.functional.gumbel_softmax(s, tau=1e-1, hard=True, dim=2)
        # debug: suppose always the last arrangement is considered
        # p = torch.zeros_like(s)
        # p[:, :, -1, :] = 1

        hbar = torch.sum(h.view(B, L, N, -1) * p, 2)
        hbar = self.inside_normalize_func(hbar)
        sbar = torch.sum(s * p, 2)

        inside_fill_chart(batch_info, chart, index, hbar, sbar)

        # use the weight to build an argmax tree for each cell
        self.decode_inside_tree(batch_info.level, p)

    def decode_inside_tree(self, level, p):
        """
        This method is meant to be private, and should not be overriden.
        Instead, override `inside_hook`.
        """
        if level == 0:
            return

        length = self.length
        B = self.batch_size
        L = length - level

        component_lookup = build_inside_component_lookup(self.index, BatchInfo(length=length, level=level))
        argmax = p.argmax(dim=2)
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

