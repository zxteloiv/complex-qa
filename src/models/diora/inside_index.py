import torch

from .offset_cache import get_offset_cache


class InsideIndex(object):
    def get_pairs(self, level, i):
        pairs = []
        for constituent_num in range(0, level):
            l_level = constituent_num
            l_i = i - level + constituent_num
            r_level = level - 1 - constituent_num
            r_i = i
            pair = ((l_level, l_i), (r_level, r_i))
            pairs.append(pair)
        return pairs

    def get_all_pairs(self, level, n):
        pairs = []
        for i in range(level, n):
            pairs += self.get_pairs(level, i)
        return pairs


def get_inside_components(length, level, offset_cache=None):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = InsideIndex()
    pairs = index.get_all_pairs(level, length)

    L = length - level
    n_constituents = len(pairs) // L
    output = []

    for i in range(n_constituents):
        index_l, index_r = [], []
        span_x, span_l, span_r = [], [], []

        l_level = i
        r_level = level - l_level - 1

        l_start = 0
        l_end = L

        r_start = length - L - r_level
        r_end = length - r_level

        if l_level < 0:
            l_level = length + l_level
        if r_level < 0:
            r_level = length + r_level

        # The span being targeted.
        for pos in range(l_start, l_end):
            span_x.append((level, pos))

        # The left child.
        for pos in range(l_start, l_end):
            offset = offset_cache[l_level]
            idx = offset + pos
            index_l.append(idx)
            span_l.append((l_level, pos))

        # The right child.
        for pos in range(r_start, r_end):
            offset = offset_cache[r_level]
            idx = offset + pos
            index_r.append(idx)
            span_r.append((r_level, pos))

        output.append((index_l, index_r, span_x, span_l, span_r))

    return output


def build_inside_component_lookup(index, batch_info):
    offset_cache = index.get_offset(batch_info.length)
    components = get_inside_components(batch_info.length, batch_info.level, offset_cache)

    component_lookup = {}
    for idx, (_, _, x_span, l_span, r_span) in enumerate(components):
        for j, (x_level, x_pos) in enumerate(x_span):
            l_level, l_pos = l_span[j]
            r_level, r_pos = r_span[j]
            component_lookup[(x_pos, idx)] = (l_level, l_pos, r_level, r_pos)
    return component_lookup


def get_inside_index(length, level, offset_cache=None, cuda=False):
    components = get_inside_components(length, level, offset_cache)

    idx_l, idx_r = [], []

    for i, (index_l, index_r, _, _, _) in enumerate(components):
        idx_l.append(index_l)
        idx_r.append(index_r)

    device = torch.cuda.current_device() if cuda else None
    idx_l = torch.tensor(idx_l, dtype=torch.int64, device=device).transpose(0, 1).contiguous().flatten()
    idx_r = torch.tensor(idx_r, dtype=torch.int64, device=device).transpose(0, 1).contiguous().flatten()

    return idx_l, idx_r

