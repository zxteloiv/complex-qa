import torch
import torch.nn as nn

from scipy.special import factorial

from .outside_index import get_outside_index, get_topk_outside_index
from .inside_index import get_inside_index
from .offset_cache import get_offset_cache


class UnitNorm(object):
    def __call__(self, x, p=2, eps=1e-8):
        return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


class NormalizeFunc(nn.Module):
    def __init__(self, mode='none'):
        super(NormalizeFunc, self).__init__()
        self.mode = mode

    def forward(self, x):
        mode = self.mode
        if mode == 'none':
            return x
        elif mode == 'unit':
            return UnitNorm()(x)
        elif mode == 'layer':
            return nn.functional.layer_norm(x, x.shape[-1:])
        raise Exception('Bad mode = {}'.format(mode))


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
