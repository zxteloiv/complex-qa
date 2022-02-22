import collections

import numpy as np

import torch
import torch.nn as nn

from base_model import DioraBase
from diora import Bilinear
from diora import ComposeMLP

from net_utils import build_chart
from net_utils import BatchInfo
from net_utils import inside_fill_chart, outside_fill_chart
from net_utils import get_inside_states
from inside_index import build_inside_component_lookup
from outside_index import get_outside_components


def get_inside_chart_cfg(diora: 'DioraMLPWithTopk', level=None, K=None, device=None):
    inputs = {}
    inputs['device'] = device
    inputs['batch_size'] = diora.batch_size
    inputs['length'] = diora.length
    inputs['size'] = diora.size
    inputs['index'] = diora.index
    inputs['score_func'] = diora.inside_score_func
    inputs['compose_func'] = diora.inside_compose_func
    inputs['normalize_func'] = diora.inside_normalize_func
    inputs['training'] = diora.training
    inputs['charts'] = diora.charts

    cfg = {'topk': K, 'mode': 'inside', 'level': level}
    return inputs, cfg


def get_outside_chart_cfg(diora: 'DioraMLPWithTopk', level=None, K=None, device=None):
    inputs = {}
    inputs['device'] = device
    inputs['batch_size'] = diora.batch_size
    inputs['length'] = diora.length
    inputs['size'] = diora.size
    inputs['inside_charts'] = diora.charts
    inputs['root_vector_out_h'] = diora.root_vector_out_h
    inputs['outside_normalize_func'] = diora.outside_normalize_func
    inputs['index'] = diora.index
    inputs['score_func'] = diora.outside_score_func
    inputs['compose_func'] = diora.outside_compose_func
    inputs['normalize_func'] = diora.outside_normalize_func
    inputs['training'] = diora.training
    inputs['outside_charts'] = diora.charts
    cfg = {'topk': K, 'mode': 'outside', 'level': level}
    return inputs, cfg


class ChartUtil(nn.Module):
    def __init__(self, topk, mode, level):
        super(ChartUtil, self).__init__()
        self.topk = topk
        self.mode = mode
        self.level = level

    def run(self, inputs):
        if self.mode == 'inside':
            return self.f_inside(inputs)
        elif self.mode == 'outside':
            return self.f_outside(inputs)

    def f_inside(self, inputs):
        outputs = {'by_level': {}}
        level = self.level
        batch_info = BatchInfo(batch_size=inputs['batch_size'], length=inputs['length'],
                               size=inputs['size'], level=level, phase='inside')
        outputs['by_level'][level] = self.f_hard_inside_helper(inputs, batch_info)

        return outputs

    def f_outside(self, inputs):
        outputs = {'by_level': {}}
        outputs['by_level'][self.level] = self.f_hard_outside_helper(inputs)
        return outputs

    @staticmethod
    def get_tensor_product_mask(K):
        l_prod = []
        for i0 in range(K):
            lst = tuple(1 if i0 == i1 else 0 for i1 in range(K) for i2 in range(K))
            l_prod.append(lst)
        l_prod = tuple(l_prod)

        r_prod = []
        for i0 in range(K):
            lst = tuple(1 if i0 == i2 else 0 for i1 in range(K) for i2 in range(K))
            r_prod.append(lst)
        r_prod = tuple(r_prod)

        pairs = np.concatenate([
            np.array(l_prod).argmax(axis=0).reshape(1, -1),
            np.array(r_prod).argmax(axis=0).reshape(1, -1)
        ], axis=0).T.tolist()

        return l_prod, r_prod, pairs

    @staticmethod
    def outside_convert_to_idx(n_idx, pk, sk, K):
        return n_idx * K * K + pk * K + sk

    def f_hard_inside_helper(self, inputs, batch_info):
        # Note: Outputs should be read-only.
        device = inputs['device']
        B = inputs['batch_size']
        L = inputs['length'] - self.level
        N = self.level
        size = inputs['size']
        batch_size = inputs['batch_size']

        index = inputs['index']
        charts = inputs['charts']
        CH = {}
        compose_func = inputs['compose_func']
        normalize_func = inputs['normalize_func']

        K = self.topk

        component_lookup = build_inside_component_lookup(index, batch_info)

        # DIORA.
        assert len(charts) == K
        for i, chart in enumerate(charts):
            lh, rh = get_inside_states(batch_info, chart['inside_h'], index, batch_info.size)
            CH.setdefault('lh', []).append(lh)
            CH.setdefault('rh', []).append(rh)

            ls, rs = get_inside_states(batch_info, chart['inside_s'], index, 1)
            CH.setdefault('ls', []).append(ls)
            CH.setdefault('rs', []).append(rs)

        # All Combos
        mat = inputs['score_func'].mat

        # a : (B, L, N, K, D)
        # b : (B, L, N, D, K)
        # out : (B, L, N, K, K)
        lh = torch.cat([CH['lh'][i].view(B, L, N, 1, size) for i in range(K)], 3)
        rh = torch.cat([CH['rh'][i].view(B, L, N, size, 1) for i in range(K)], 4)
        s_raw = torch.matmul(torch.matmul(lh, mat), rh)     # a bilinear layer for the states

        l_prod, r_prod, pairs = self.get_tensor_product_mask(K)
        select_ls = torch.tensor(l_prod, dtype=torch.float, device=device)
        select_rs = torch.tensor(r_prod, dtype=torch.float, device=device)
        # a : (B, L, N, K)
        # b : (B, L, N, K)
        ls = torch.cat([CH['ls'][i].view(B, L, N, 1) for i in range(K)], 3)
        rs = torch.cat([CH['rs'][i].view(B, L, N, 1) for i in range(K)], 3)
        combo_ls = torch.matmul(ls, select_ls).view(B, L, N * K * K, 1)
        combo_rs = torch.matmul(rs, select_rs).view(B, L, N * K * K, 1)
        combo_s = s_raw.view(B, L, N * K * K, 1)
        s = combo_s + combo_ls + combo_rs

        # We should not include any split that includes an in-complete beam.
        def penalize_incomplete_splits():
            force_index = collections.defaultdict(list)
            for i_b in range(batch_size):
                for pos in range(L):
                    for i, (l_k, r_k) in enumerate(pairs):
                        idx = i // (K**2)
                        l_level, l_pos, r_level, r_pos = component_lookup[(pos, idx)]
                        max_possible_trees = index.get_catalan(l_level + 1)
                        if max_possible_trees < l_k + 1:
                            force_index['i_b'].append(i_b)
                            force_index['pos'].append(pos)
                            force_index['i'].append(i)
                        max_possible_trees = index.get_catalan(r_level + 1)
                        if max_possible_trees < r_k + 1:
                            force_index['i_b'].append(i_b)
                            force_index['pos'].append(pos)
                            force_index['i'].append(i)
            s[force_index['i_b'], force_index['pos'], force_index['i']] = -1e8

        # Don't do this when the trees are specified since it's possible for duplicate
        # sub-trees to exist.
        penalize_incomplete_splits()

        # Aggregate.
        topk_s, topk_idx = s.topk(dim=2, k=K)

        # Force topk according to given constraints.
        def _dense_compose():
            lh = torch.cat([CH['lh'][i].view(B, L, N, 1, size) for i in range(K)], 3)
            rh = torch.cat([CH['rh'][i].view(B, L, N, 1, size) for i in range(K)], 3)

            combo_lh = torch.einsum('blnkd,kz->blnzd', lh, select_ls).contiguous().view(-1, size)
            combo_rh = torch.einsum('blnkd,kz->blnzd', rh, select_rs).contiguous().view(-1, size)
            all_h = compose_func([combo_lh, combo_rh]).view(B, L, N * K * K, size)

            topk_h = all_h.gather(index=topk_idx.expand(B, L, K, size), dim=2)
            topk_h = normalize_func(topk_h)

            all_idx = torch.tensor(range(N * K * K), dtype=torch.long, device=device)\
                .view(1, 1, N * K * K, 1).expand(B, L, N * K * K, 1)

            # Apply topk constraints.
            # TODO: This should be done w/o for loop.
            topk_n_idx = all_idx // (K**2)
            topk_lk = all_idx % (K**2) // K
            topk_rk = all_idx % (K**2) % K

            is_valid = []
            for i_b in range(B):
                for pos in range(L):
                    for n_idx, lk, rk in zip(topk_n_idx[i_b, pos].view(-1).tolist(),
                                             topk_lk[i_b, pos].view(-1).tolist(),
                                             topk_rk[i_b, pos].view(-1).tolist()):

                        l_level, l_pos, r_level, r_pos = component_lookup[(pos, n_idx)]

                        num_components = l_level
                        max_possible_trees = index.get_catalan(num_components + 1)
                        if max_possible_trees < lk + 1:
                            is_valid.append(False)
                            continue

                        num_components = r_level
                        max_possible_trees = index.get_catalan(num_components + 1)
                        if max_possible_trees < rk + 1:
                            is_valid.append(False)
                            continue

                        is_valid.append(True)
            is_valid = torch.tensor(is_valid, dtype=torch.bool, device=device).view(B, L, N * K * K, 1)
            all_idx = all_idx[is_valid].view(B, L, -1, 1)
            all_h = all_h[is_valid.expand(B, L, N*K*K, size)].view(B, L, -1, size)

            return all_idx, all_h, topk_h

        all_idx, all_h, topk_h = _dense_compose()

        assert topk_s.shape == (B, L, K, 1), topk_s.shape

        # We should not add more than the number of possible trees to the beam.
        max_possible_trees = index.get_catalan(N + 1)
        if max_possible_trees < K:
            topk_s[:, :, max_possible_trees:] = -1e8

        topk_n_idx = topk_idx // (K**2)
        topk_lk = topk_idx % (K**2) // K
        topk_rk = topk_idx % (K**2) % K

        # Result.
        result = {}
        for i in range(K):
            result.setdefault('h', []).append(topk_h[:, :, i])
            result.setdefault('s', []).append(topk_s[:, :, i])

        result['topk_h'] = topk_h
        result['topk_s'] = topk_s
        result['topk_n_idx'] = topk_n_idx
        result['topk_lk'] = topk_lk
        result['topk_rk'] = topk_rk

        return result

    def f_hard_outside_helper(self, inputs):
        level = self.level
        device = inputs['device']
        B = inputs['batch_size']
        L = inputs['length'] - level
        N = inputs['length'] - level - 1
        K = self.topk
        size = inputs['size']
        length = inputs['length']
        index = inputs['index']
        icharts = inputs['inside_charts']
        ocharts = inputs['outside_charts']
        CH = {}
        compose_func = inputs['compose_func']
        normalize_func = inputs['normalize_func']

        # Incorporate elmo.
        offset_cache = index.get_offset(length)
        components = get_outside_components(length, level, offset_cache)

        component_lookup = {}
        for p_k in range(K):
            for s_k in range(K):
                for i, (p_span, s_span) in enumerate(components):
                    p_level, p_pos = p_span
                    s_level, s_pos = s_span
                    idx = i // L
                    x_pos = i % L
                    component_lookup[(x_pos, idx, p_k, s_k)] = (p_level, p_pos, s_level, s_pos)

        # DIORA.
        ## 0
        p_prod, s_prod, pairs = ChartUtil.get_tensor_product_mask(K)
        p_index, p_info, s_index, s_info = index.get_topk_outside_index(length, level, K)

        def get_outside_states(pchart, schart, size):
            ps = pchart.index_select(index=p_index, dim=1).view(-1, size)
            ss = schart.index_select(index=s_index, dim=1).view(-1, size)
            return ps, ss

        for i in range(K):
            ph, sh = get_outside_states(ocharts[i]['outside_h'], icharts[i]['inside_h'], size)
            CH.setdefault('ph', []).append(ph)
            CH.setdefault('sh', []).append(sh)

            ps, ss = get_outside_states(ocharts[i]['outside_s'], icharts[i]['inside_s'], 1)
            CH.setdefault('ps', []).append(ps)
            CH.setdefault('ss', []).append(ss)

        ## All Combos
        mat = inputs['score_func'].mat

        ph = torch.cat([CH['ph'][i].view(B, L, N, 1, size) for i in range(K)], 3)
        sh = torch.cat([CH['sh'][i].view(B, L, N, size, 1) for i in range(K)], 4)
        s_raw = torch.matmul(torch.matmul(ph, mat), sh)
        select_ps = torch.tensor(p_prod, dtype=torch.float, device=device)
        select_ss = torch.tensor(s_prod, dtype=torch.float, device=device)
        ps = torch.cat([CH['ps'][i].view(B, L, N, 1) for i in range(K)], 3)
        ss = torch.cat([CH['ss'][i].view(B, L, N, 1) for i in range(K)], 3)
        combo_ps = torch.matmul(ps, select_ps).view(B, L, N * K * K, 1)
        combo_ss = torch.matmul(ss, select_ss).view(B, L, N * K * K, 1)
        combo_s = s_raw.view(B, L, N * K * K, 1)
        s = combo_s + combo_ps + combo_ss
        s_reshape = s
        assert s_reshape.shape == (B, L, N * K * K, 1)

        # We should not include any split that includes an in-complete beam.
        def penalize_incomplete_splits():
            force_index = collections.defaultdict(list)
            for i_b in range(B):
                for pos in range(L):
                    for n_idx in range(N):
                        for p_k, s_k in pairs:
                            new_idx = ChartUtil.outside_convert_to_idx(n_idx, p_k, s_k, K)
                            p_level, p_pos, s_level, s_pos = component_lookup[(pos, n_idx, p_k, s_k)]
                            p_num_components = length - p_level - 1
                            s_num_components = s_level
                            max_possible_trees = index.get_catalan(p_num_components + 1)
                            if max_possible_trees < p_k + 1:
                                force_index['i_b'].append(i_b)
                                force_index['pos'].append(pos)
                                force_index['new_idx'].append(new_idx)

                            max_possible_trees = index.get_catalan(s_num_components + 1)
                            if max_possible_trees < s_k + 1:
                                force_index['i_b'].append(i_b)
                                force_index['pos'].append(pos)
                                force_index['new_idx'].append(new_idx)

            s[force_index['i_b'], force_index['pos'], force_index['new_idx']] = -1e8

        # Don't do this when the trees are specified since it's possible for duplicate
        # sub-trees to exist.
        penalize_incomplete_splits()

        # Aggregate.
        topk_s, topk_idx = s_reshape.topk(dim=2, k=K)

        # Selective compose.
        topk_n_idx = topk_idx // (K**2)
        topk_pk = topk_idx % (K**2) // K
        topk_sk = topk_idx % (K**2) % K
        # Also can be computed as:
        # n_idx = topk_idx // (K * K) # changes once per K**2 steps
        # p_k = topk_idx // K % K # changes once per K steps
        # s_k = topk_idx % K # changes every step

        topk_p_idx = topk_n_idx * K + topk_pk
        sel_ph = ph.view(B, L, N * K, size).gather(dim=2, index=topk_p_idx.expand(B, L, K, size)).view(-1, size)

        topk_s_idx = topk_n_idx * K + topk_sk
        # TODO: If possible, remove this transpose. Although this approach still has less transpose than previously.
        sel_sh = sh.transpose(3, 4).reshape(B, L, N * K, size).gather(dim=2, index=topk_s_idx.expand(B, L, K, size)).view(-1, size)

        topk_h = compose_func([sel_ph, sel_sh]).view(B, L, K, size)
        topk_h = normalize_func(topk_h)

        # We should not add more than the number of possible trees to the beam.
        max_possible_trees = index.get_catalan(N + 1)
        if max_possible_trees < K:
            topk_s[:, :, max_possible_trees:] = -1e8

        # Result.
        result = {}
        assert topk_h.shape[2] == K
        assert topk_s.shape[2] == K
        for i_k in range(K):
            result.setdefault('h', []).append(topk_h[:, :, i_k])
            result.setdefault('s', []).append(topk_s[:, :, i_k])

        result['topk_h'] = topk_h
        result['topk_s'] = topk_s
        result['topk_n_idx'] = topk_n_idx
        result['topk_pk'] = topk_pk
        result['topk_sk'] = topk_sk

        return result


class DioraMLPWithTopk(DioraBase):
    def __init__(self, *args, **kwargs):
        # {'size': 400, 'outside': True, 'normalize': 'unit',
        # 'n_layers': 2, 'K': 3, 'projection_layer': None}
        self.charts = None
        self.n_layers = kwargs.get('n_layers', None)
        super(DioraMLPWithTopk, self).__init__(*args, **kwargs)
        self.K = kwargs.get('K', 2)

    def reset(self):
        super(DioraMLPWithTopk, self).reset()
        if self.charts is not None:
            for i in range(1, self.K):
                chart = self.charts[i]
                keys = list(chart.keys())
                for k in keys:
                    self.nested_del(chart, k)
        self.charts = None

    def init_parameters(self):
        # Model parameters for transformation required at both input and output
        self.inside_score_func = Bilinear(self.size)
        self.inside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers)
        self.outside_score_func = Bilinear(self.size)
        self.outside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers)
        self.root_vector_out_h = nn.Parameter(torch.FloatTensor(self.size))

    def inside_func(self, batch_info):
        device = self.device
        L = batch_info.length - batch_info.level
        level = batch_info.level
        K = self.K
        index = self.index

        inputs, cfg = get_inside_chart_cfg(self, level=level, K=K, device=device)

        chart_output = ChartUtil(**cfg).run(inputs)['by_level'][level]

        topk_n_idx = chart_output['topk_n_idx']
        topk_lk = chart_output['topk_lk']
        topk_rk = chart_output['topk_rk']

        for i in range(self.K):
            inside_fill_chart(batch_info, self.charts[i], index, chart_output['h'][i], chart_output['s'][i])

        # backtrack
        component_lookup = build_inside_component_lookup(index, batch_info)
        for i_b in range(self.batch_size):
            for pos in range(L):
                for i_k in range(K):
                    n_idx = topk_n_idx[i_b, pos, i_k].item()
                    l_k = topk_lk[i_b, pos, i_k].item()
                    r_k = topk_rk[i_b, pos, i_k].item()
                    l_level, l_pos, r_level, r_pos = component_lookup[(pos, n_idx)]
                    self.cache['inside_tree'][(i_b, i_k)][(level, pos)] = \
                        self.cache['inside_tree'][(i_b, l_k)][(l_level, l_pos)] + \
                        self.cache['inside_tree'][(i_b, r_k)][(r_level, r_pos)] + \
                        [(level, pos)]

    def outside_func(self, batch_info):
        level = batch_info.level

        inputs, cfg = get_outside_chart_cfg(self, level=batch_info.level, K=self.K, device=self.device)
        chart_output = ChartUtil(**cfg).run(inputs)['by_level'][level]
        for i in range(self.K):
            outside_fill_chart(batch_info, self.charts[i], self.index, chart_output['h'][i], chart_output['s'][i])

    def initialize(self, x):
        result = super(DioraMLPWithTopk, self).initialize(x)
        batch_size, length = x.shape[:2]
        size = self.size
        charts = [self.chart]
        for _ in range(1, self.K):
            charts.append(build_chart(batch_size, length, size, dtype=torch.float, cuda=self.is_cuda))
        # Initialize outside root.
        h = self.root_vector_out_h.view(1, 1, size).expand(batch_size, 1, size)
        h = self.outside_normalize_func(h)
        for i in range(0, self.K):
            if i == 0:
                continue
            # Never should be selected.
            charts[i]['inside_s'][:] = -1e8
            charts[i]['outside_s'][:] = -1e8
            charts[i]['outside_h'][:, -1:] = h
        self.charts = charts
        return result

