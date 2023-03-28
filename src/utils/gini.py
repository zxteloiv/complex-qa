import torch
from math import prod


def slow_gini_index(weights: torch.Tensor, lengths: torch.LongTensor):
    """

    :param weights: (b, *, max_num_classes)
    :param lengths: (b,) indicating valid num_classes
    :return:
    """
    old_size = weights.size()  # (b, *, num_classes)
    row_len = prod(old_size[1:-1])
    rw = weights.reshape(-1, old_size[-1])  # (-1, num_classes)
    sw, _ = rw.sort(dim=-1)

    def gini_1d(w: torch.Tensor) -> torch.Tensor:
        acc: torch.Tensor = 0  # noqa
        N = w.size(0)  # w is a 1-d vector
        for k, c in enumerate(w):
            acc += c * (N - (k + 1) + .5) / N
        return 1 - 2 * acc

    indices = torch.stack([gini_1d(row[-lengths[i // row_len]:]) for i, row in enumerate(sw)])
    return indices.reshape(old_size[:-1])  # (b, *)
