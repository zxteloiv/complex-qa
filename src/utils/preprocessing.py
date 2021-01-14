from collections import defaultdict

def nested_list_numbers_to_tensors(nested: list, padding=0, example=None):
    import torch
    """Turn a list of list of list of list ... of integers to a tensor with the given padding"""
    ndim_max = defaultdict(lambda: 0)

    def _count_nested_max(nested, depth):
        if type(nested) not in (list, tuple, set):
            return

        ndim_max[depth] = max(ndim_max[depth], len(nested))
        for x in nested:
            _count_nested_max(x, depth + 1)

    _count_nested_max(nested, 0)
    ndim_max = [ndim_max[d] for d in sorted(ndim_max.keys())]

    def _get_padding_at_depth(depth):
        size = ndim_max[depth:]
        lump = padding
        for i in reversed(size):
            lump = [lump] * i
        return lump

    def _pad_nested(nested, depth):
        if type(nested) not in (list, tuple, set):
            return nested

        if len(nested) < ndim_max[depth]:
            nested = list(nested) + [_get_padding_at_depth(depth + 1)] * (ndim_max[depth] - len(nested))

        return [_pad_nested(x, depth + 1) for x in nested]

    full_fledged = _pad_nested(nested, 0)
    dev = dtype = None
    if example is not None:
        dev = example.device
        dtype = example.dtype

    return torch.tensor(full_fledged, device=dev, dtype=dtype)


