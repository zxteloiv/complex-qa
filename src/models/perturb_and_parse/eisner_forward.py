# the forward pass of the Eisner's algorithm, built upon the functions of pytorch builtins,
# in this way we do not have to implement the backward pass computations, but the backward
# may be significantly slow than an inherited Function with customized backward pass.
#
# the algorithm is based on
# DIFFERENTIABLE PERTURB - AND - PARSE :
# SEMI-SUPERVISED PARSING WITH A STRUCTURED VARIATIONAL AUTOENCODER
# https://openreview.net/forum?id=BJlgNh0qKQ
#


import torch
from enum import IntEnum


class CTag(IntEnum):
    R_COMP = 0
    L_COMP = 1
    R_INCOMP = 2
    L_INCOMP = 3


def eisner(weights: torch.Tensor, mask: torch.Tensor, hard: bool = False) -> torch.Tensor:
    """
    Forward of the eisner algorithm.
    :param weights: (batch, seq_len, seq_len)
    :param mask: (batch, seq_len)
    :param hard: True to turn the softmax of span splits into argmax
    :return:
    """
    batch, seq_len = weights.size()[:2]
    cw = weights.new_zeros((4, batch, seq_len, seq_len))            # chart weights
    bp = weights.new_zeros((4, batch, seq_len, seq_len, seq_len))   # backtrack pointer

    def inside():
        for len in range(1, seq_len):
            for i in range(seq_len - len):
                _fill_charts(i, i + len)

    def _softmax(x):
        if not hard:
            return torch.softmax(x, dim=-1)
        else:  # doesn't support backpropagation when hard=True
            return torch.zeros_like(x).scatter_(-1, x.argmax(-1).unsqueeze(-1), 1)

    def _fill_charts(i, j):
        # r/l: right or left;
        # c/i: complete or incomplete
        # bp: backtrack pointer
        r_c, l_c, r_i, l_i = cw[CTag.R_COMP], cw[CTag.L_COMP], cw[CTag.R_INCOMP], cw[CTag.L_INCOMP]
        bp_r_c, bp_l_c, bp_r_i, bp_l_i = bp[CTag.R_COMP], bp[CTag.L_COMP], bp[CTag.R_INCOMP], bp[CTag.L_INCOMP]

        # incomplete cells, (
        right_i = r_c[:, i, i:j] + l_c[:, i + 1:j + 1, j] + weights[:, j, i]
        left_i = r_c[:, i, i:j] + l_c[:, i + 1:j + 1, j] + weights[:, i, j]
        bp_right_i = _softmax(right_i)
        bp_left_i = _softmax(left_i)

        r_i[:, i, j] = (right_i * bp_right_i).sum(-1)
        l_i[:, i, j] = (left_i * bp_left_i).sum(-1)
        bp_r_i[:, i, j, i:j] = bp_right_i
        bp_l_i[:, i, j, i:j] = bp_left_i

        # complete chart cells, (batch, j - i)
        left_c = l_c[:, i, i:j] + l_i[:, i:j, j]
        right_c = r_i[:, i, i + 1:j + 1] + r_c[:, i + 1:j + 1, j]
        bp_left_c = _softmax(left_c)
        bp_right_c = _softmax(right_c)

        l_c[:, i, j] = (bp_left_c * left_c).sum(-1)
        r_c[:, i, j] = (bp_right_c * right_c).sum(-1)
        bp_l_c[:, i, j, i:j] = bp_left_c
        bp_r_c[:, i, j, i+1:j+1] = bp_right_c

    inside()

    cb = weights.new_zeros(4, batch, seq_len, seq_len)   # contrib
    cb[CTag.R_COMP, list(range(batch)), 0, mask.sum(-1)] = 1

    def backptr():
        pass

    pass

