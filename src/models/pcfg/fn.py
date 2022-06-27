from typing import Union, List

import torch


def unit_norm(x, p=2, eps=1e-12):
    return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


def _un(x: torch.Tensor, dims: Union[List[int], int]):
    if isinstance(dims, int):
        return x.unsqueeze(dims)
    else:
        for d in sorted(dims):
            x = x.unsqueeze(d)
        return x