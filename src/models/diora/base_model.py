import torch
import torch.nn as nn
from .net_utils import NormalizeFunc, BatchInfo, build_chart, Index, Bilinear, ComposeMLP


class SigmaH(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.sqrt(torch.reciprocal(1 + torch.exp(-x))))


class DioraBase(nn.Module):
    r"""DioraBase"""
    def safe_set_K(self, val):
        self.reset()
        self.K = val

    def __init__(self, input_size=None, size=None, outside=True, **kwargs):
        super(DioraBase, self).__init__()
        self.K = 1
        self.size = size
        self.input_size = input_size or size
        self.default_outside = outside
        self.inside_normalize_func = NormalizeFunc('layer', self.size)
        self.outside_normalize_func = NormalizeFunc('layer', self.size)

        # self.activation = nn.Mish()
        self.activation = SigmaH()
        # self.activation = nn.Tanh()
        self.leaf_linear = nn.Linear(self.input_size, self.size)
        self.root_vector_out_h = nn.Parameter(torch.zeros(self.size))
        self.index = None
        self.cache = None
        self.chart = None

        self.n_layers = kwargs.get('n_layers', 2)
        self.inside_score_func = Bilinear(self.size)
        self.inside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers)
        self.outside_score_func = Bilinear(self.size)
        self.outside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers)

        self.reset()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        device = self.device
        return device.index is not None and device.index >= 0

    @property
    def inside_h(self):
        return self.chart['inside_h']

    @property
    def inside_s(self):
        return self.chart['inside_s']

    @property
    def outside_h(self):
        return self.chart['outside_h']

    @property
    def outside_s(self):
        return self.chart['outside_s']

    def cuda(self, device=None):
        super(DioraBase, self).cuda(device)
        if self.index is not None:
            self.index.cuda = True

    def get(self, chart, level):
        length = self.length
        L = length - level
        offset = self.index.get_offset(length)[level]
        return chart[:, offset:offset+L]

    def leaf_transform(self, x):
        normalize_func = self.inside_normalize_func

        h = self.activation(self.leaf_linear(x))

        input_shape = x.shape[:-1]
        h = normalize_func(h.view(*input_shape, self.size))

        return h

    def inside_pass(self):
        # span length from 1 up to max length
        for level in range(1, self.length):
            batch_info = BatchInfo(
                batch_size=self.batch_size,
                length=self.length,
                size=self.size,
                level=level,
                phase='inside',
                )
            self.inside_func(batch_info)

    def inside_func(self, batch_info):
        raise NotImplementedError

    def initialize_outside_root(self):
        B = self.batch_size
        D = self.size
        normalize_func = self.outside_normalize_func

        h = self.root_vector_out_h.view(1, 1, D).expand(B, 1, D)
        h = normalize_func(h)
        self.outside_h[:, -1:] = h

    def outside_pass(self):
        self.initialize_outside_root()

        for level in range(self.length - 2, -1, -1):
            batch_info = BatchInfo(
                batch_size=self.batch_size,
                length=self.length,
                size=self.size,
                level=level,
                phase='outside',
                )

            self.outside_func(batch_info)

    def outside_func(self, batch_info):
        raise NotImplementedError

    def init_with_batch(self, h, info=None):
        info = info or dict()
        self.batch_size, self.length, _ = h.shape
        self.outside = info.get('outside', self.default_outside)
        # the chart size is (batch, num_chart_cells, size)
        # set the leaf charts to the transformed hidden states
        self.inside_h[:, :self.length] = h
        self.cache['inside_tree'] = {}
        for i in range(self.batch_size):
            for i_k in range(self.K):
                tree = {}
                level = 0   # lowest or substring length ? every cell up to length is set to empty
                for pos in range(self.length):
                    tree[(level, pos)] = []
                self.cache['inside_tree'][(i, i_k)] = tree

    def nested_del(self, o, k):
        if isinstance(o[k], dict):
            keys = list(o[k].keys())
            for kk in keys:
                self.nested_del(o[k], kk)
        del o[k]

    def reset(self):
        self.batch_size = None
        self.length = None

        if self.chart is not None:
            keys = list(self.chart.keys())
            for k in keys:
                self.nested_del(self.chart, k)
        self.chart = None

        if self.cache is not None:
            keys = list(self.cache.keys())
            for k in keys:
                self.nested_del(self.cache, k)
        self.cache = None

    def initialize(self, x):
        size = self.size
        batch_size, length = x.shape[:2]
        self.chart = build_chart(batch_size, length, size, dtype=torch.float, cuda=self.is_cuda)
        self.cache = {}

    def forward(self, x, info=None):
        # info: dict_keys(['inside_pool', 'outside', 'raw_parse',
        # 'constituency_tags', 'pos_tags', 'binary_tree', 'example_ids'])
        if self.index is None:
            self.index = Index(cuda=self.is_cuda)

        self.reset()
        self.initialize(x)

        # leaf linear + tanh transform + normalize to unit length
        h = self.leaf_transform(x)

        self.init_with_batch(h, info)
        self.inside_pass()

        if self.outside:
            self.outside_pass()

        return self.chart

    @classmethod
    def from_kwargs_dict(cls, kwargs_dict):
        return cls(**kwargs_dict)

