import torch
import torch.nn as nn

from scipy.special import factorial

from .outside_index import get_outside_index, get_topk_outside_index
from .inside_index import get_inside_index
from .offset_cache import get_offset_cache
from ..modules.decomposed_bilinear import DecomposedBilinear


class UnitNorm(object):
    def __call__(self, x, p=2, eps=1e-8):
        return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


class NormalizeFunc(nn.Module):
    def __init__(self, mode='none', size=None):
        super(NormalizeFunc, self).__init__()
        self.mode = mode
        if mode == 'layer':
            self.norm_func = nn.LayerNorm(size)
        elif mode == 'unit':
            self.norm_func = UnitNorm()
        elif mode == 'none':
            self.norm_func = lambda x: x
        else:
            raise ValueError('Bad mode = {}'.format(mode))

    def forward(self, x):
        return self.norm_func(x)


# Composition Functions
class ComposeMLP(nn.Module):
    def __init__(self, size, activation, n_layers=2):
        super(ComposeMLP, self).__init__()
        assert n_layers > 0

        self.size = size
        self.activation = activation
        self.n_layers = n_layers

        self.layers = torch.nn.ModuleList([
            nn.Linear(2 * self.size if n == 0 else self.size, self.size)
            for n in range(n_layers)
        ])

    def forward(self, hs):
        h = torch.cat(hs, 1)
        for l in self.layers:
            h = self.activation(l(h))
        return h


class ComposeBilinear(nn.Module):
    def __init__(self, size, activation=None, n_layers=None):
        super().__init__()
        self.bilinear = DecomposedBilinear(size, size, size,
                                           decomposed_rank=1, pool_size=size,
                                           use_linear=True, use_bias=True,
                                           )
        self.activation = activation

    def forward(self, hs):
        h = self.bilinear(hs[0], hs[1])
        if self.activation is not None:
            h = self.activation(h)
        return h


class ComposeGRU(nn.Module):
    def __init__(self, size, activation=None, n_layers=None):
        super().__init__()
        self.gru_cell = nn.GRUCell(size, size)

    def forward(self, hs):
        h, x = hs[0], hs[1]
        return self.gru_cell(x, h)


composer_name_to_cls = {
    'mlp': ComposeMLP,
    'gru': ComposeGRU,
    'bilinear': ComposeBilinear,
}


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


class BatchInfo(object):
    def __init__(self, **kwargs):
        super(BatchInfo, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_chart(batch_size, length, size, dtype=None, cuda=False):
    # triangle area: length * (length + 1) / 2 = 55 if length = 10
    ncells = int(length * (1 + length) / 2)
    device = torch.cuda.current_device() if cuda else None

    chart = {
        # inside
        'inside_h': torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device),
        'inside_s': torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device),
        # outside
        'outside_h': torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device),
        'outside_s': torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)
    }

    return chart


def get_catalan(n):
    if n > 10:
        return 5000 # HACK: We only use this to check number of trees, and this avoids overflow.
    n = n - 1
    def choose(n, p): return factorial(n) / (factorial(p) * factorial(n-p))
    return int(choose(2 * n, n) // (n + 1))


class Index(object):
    def __init__(self, cuda=False, enable_caching=True):
        super(Index, self).__init__()
        self.cuda = cuda
        self.cache = {}
        self.enable_caching = enable_caching

    def cached_lookup(self, func, name, key):
        if name not in self.cache:
            self.cache[name] = {}
        cache = self.cache[name]
        if self.enable_caching:
            if key not in cache:
                cache[key] = func()
            return cache[key]
        else:
            return func()

    def get_catalan(self, n):
        name = 'catalan'
        key = n
        def func(): return get_catalan(n)
        return self.cached_lookup(func, name, key)

    def get_offset(self, length):
        name = 'offset_cache'
        key = length
        def func(): return get_offset_cache(length)
        return self.cached_lookup(func, name, key)

    def get_inside_index(self, length, level):
        name = 'inside_index_cache'
        key = (length, level)
        def func(): return get_inside_index(length, level, self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)

    def get_outside_index(self, length, level):
        name = 'outside_index_cache'
        key = (length, level)
        def func(): return get_outside_index(length, level, self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)

    def get_topk_outside_index(self, length, level, K):
        name = 'topk_outside_index_cache'
        key = (length, level, K)
        def func(): return get_topk_outside_index(length, level, K, self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)


def get_fill_chart_func(prefix):
    def fill_chart(batch_info, chart, index, h, s):
        L = batch_info.length - batch_info.level
        offset = index.get_offset(batch_info.length)[batch_info.level]
        chart[prefix+'_h'][:, offset:offset + L] = h
        chart[prefix+'_s'][:, offset:offset + L] = s
    return fill_chart


inside_fill_chart = get_fill_chart_func('inside')
outside_fill_chart = get_fill_chart_func('outside')


def get_inside_states(batch_info, chart, index, size):
    lidx, ridx = index.get_inside_index(batch_info.length, batch_info.level)

    ls = chart.index_select(index=lidx, dim=1).view(-1, size)
    rs = chart.index_select(index=ridx, dim=1).view(-1, size)

    return ls, rs


def get_outside_states(batch_info, pchart, schart, index, size):
    pidx, sidx = index.get_outside_index(batch_info.length, batch_info.level)

    ps = pchart.index_select(index=pidx, dim=1).view(-1, size)
    ss = schart.index_select(index=sidx, dim=1).view(-1, size)

    return ps, ss
