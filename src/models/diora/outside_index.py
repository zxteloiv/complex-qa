import torch

from .offset_cache import get_offset_cache


class OutsideIndex(object):
    def get_all_pairs(self, level, n):
        L = n - level   # num of cells at the level
        N = L - 1   # num of cells for upper level or the level except for one cell

        pairs = []

        for i in range(N):  # for any upper-layer cells
            jseen = 0
            for j in range(L):  # for any cells at the level
                if j < N - i:   # the current cell is on the left of the
                    s_level = n - i - 1
                    s_i = N - i - j - 1
                    p_level = s_level
                    p_i = s_level - j
                else:
                    s_level = j - 1
                    s_i = jseen
                    p_level = n - (N - s_level)
                    p_i = n - (N - s_i)
                    jseen += 1
                # pair = [(p_i, p_level), (s_i, s_level)]
                pair = [(p_level, p_i), (s_level, s_i)]
                pairs.append(pair)

        return pairs


def get_outside_components(length, level, offset_cache=None):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = OutsideIndex()
    pairs = index.get_all_pairs(level, length)
    output = []

    for pair in pairs:
        par, sis = pair
        par_lvl = par[0]
        par_pos = par[1] - par[0]
        par_span = (par_lvl, par_pos)
        sis_lvl = sis[0]
        sis_pos = sis[1] - sis[0]
        sis_span = (sis_lvl, sis_pos)

        output.append((par_span, sis_span))

    return output


def get_outside_index(length, level, offset_cache=None, cuda=False):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = OutsideIndex()
    pairs = index.get_all_pairs(level, length)

    par_lvl, par_pos = [], []
    sis_lvl, sis_pos = [], []

    for pair in pairs:
        par, sis = pair
        par_lvl.append(par[0])
        par_pos.append(par[1] - par[0])
        sis_lvl.append(sis[0])
        sis_pos.append(sis[1] - sis[0])

    device = torch.cuda.current_device() if cuda else None

    # Parent
    index = []
    for lvl, pos in zip(par_lvl, par_pos):
        offset = offset_cache[lvl]
        idx = offset + pos
        index.append(idx)
    par_index = torch.tensor(index, dtype=torch.long, device=device)

    # Sibling
    index = []
    for lvl, pos in zip(sis_lvl, sis_pos):
        offset = offset_cache[lvl]
        idx = offset + pos
        index.append(idx)
    sis_index = torch.tensor(index, dtype=torch.long, device=device)

    return par_index, sis_index


def get_topk_outside_index(length, level, K, offset_cache=None, cuda=False):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)

    L = length - level
    N = length - level - 1

    components = get_outside_components(length, level, offset_cache)

    p_info, s_info = [], []
    for i, (p_span, s_span) in enumerate(components):
        p_level, p_pos = p_span
        s_level, s_pos = s_span
        n_idx = i // L
        x_pos = i % L
        p_idx = offset_cache[p_level] + p_pos
        s_idx = offset_cache[s_level] + s_pos

        p_info.append((x_pos, n_idx, p_level, p_pos, p_idx))
        s_info.append((x_pos, n_idx, s_level, s_pos, s_idx))

    def sort_key(x):
        x_pos, n_idx, inp_level, inp_pos, inp_idx = x
        return (x_pos, n_idx)

    def get_val(x):
        x_pos, n_idx, inp_level, inp_pos, inp_idx = x
        return inp_idx

    p_info = sorted(p_info, key=sort_key)
    s_info = sorted(s_info, key=sort_key)

    device = torch.cuda.current_device() if cuda else None

    p_index = torch.tensor([get_val(x) for x in p_info], dtype=torch.long, device=device)
    s_index = torch.tensor([get_val(x) for x in s_info], dtype=torch.long, device=device)

    return p_index, p_info, s_index, s_info
