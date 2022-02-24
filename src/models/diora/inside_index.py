import torch

from .offset_cache import get_offset_cache


class InsideIndex(object):
    def get_pairs(self, level, i):
        # i - level: cell id at each level
        # for any i, the pairs length is the same
        pairs = []
        # for all previous levels, indexed by constituent_num
        for constituent_num in range(0, level):
            # left component could come from each below level
            l_level = constituent_num
            # left component id
            l_i = i - level + constituent_num
            # for any left level constituent, the right constituent is fixed
            r_level = level - 1 - constituent_num
            r_i = i
            pair = ((l_level, l_i), (r_level, r_i))
            pairs.append(pair)
        return pairs

    def get_all_pairs(self, level, n):
        pairs = []
        # at each cell of the level, w.r.t the max length n
        for i in range(level, n):
            pairs += self.get_pairs(level, i)
        return pairs


def get_inside_components(length, level, offset_cache=None):
    assert length > level >= 0, f'invalid length={length} and level={level}'
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = InsideIndex()
    pairs = index.get_all_pairs(level, length)

    L = length - level  # num of cells at the level
    n_constituents = len(pairs) // L    # each cell of the level share the same n_constituent = level
    output = []

    for i in range(n_constituents):
        index_l, index_r = [], []
        span_x, span_l, span_r = [], [], []

        l_level = i     # actually any level below is possible
        r_level = level - l_level - 1   # as long as l_level + r_level + 1 === level

        l_start = 0
        l_end = L   # num of cells at the target level

        r_start = length - L - r_level  # as long as r_end - r_start === L, the num of cells at level
        r_end = length - r_level    # num of cells at r_level

        # The span being targeted.
        for pos in range(l_start, l_end):   # pos: each cell index
            span_x.append((level, pos))     # consider all the cells at the level

        # The left child.
        for pos in range(l_start, l_end):   # pos: each cell index
            offset = offset_cache[l_level]  # for any l_level below find the offset on the chart tensor
            idx = offset + pos              # find the cell idx on the chart at the l_level
            index_l.append(idx)
            # for each cell at pos, the left span at least starts from the same column of the cell pos
            # otherwise, for pos' < pos, it shall not belong to the tree represented by the cell pos
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

    # idx_*: (sublevel, target_cells), or (constituents_num, target_cells) because n_constituents == level
    # idx_*: (target_cells * constituents,) <- (constituents_num, cell_idx_on_charts)
    #       for each cell at the target level, the idx of spans on charts
    #       where each idx entails both the child level and child pos (indicating the subtree or a span)
    device = torch.cuda.current_device() if cuda else None
    idx_l = torch.tensor(idx_l, dtype=torch.int64, device=device).transpose(0, 1).contiguous().flatten()
    idx_r = torch.tensor(idx_r, dtype=torch.int64, device=device).transpose(0, 1).contiguous().flatten()

    return idx_l, idx_r

