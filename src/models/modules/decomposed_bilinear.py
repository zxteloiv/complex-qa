from typing import Optional
import torch
from torch import nn


class DecomposedBilinear(nn.Module):
    """
    In general, the module is an approximation to the real bilinear operation with lower parameters,
    which follows the common decomposition as https://arxiv.org/abs/1805.07932

    Briefly, to approximate `aWb` where W is (left, out, right) and requires O(left * right * out) parameters,
    we decompose the W into W_o (a W_a) * (W_b b), with the parameters as
       W_a: (pool, rank, left)
       W_b: (pool, rank, right)
       W_o: (pool, output)

    where (a^T W_a) * (W_b b) is called the bilinear pool.

    The space requirement is O(pool * (rank * (left + right) + hidden))
    Under the common situation that rank ~ pool << out, the decomposed bilinear module will save much space.
    """
    def __init__(self,
                 left_size: int,
                 right_size: int,
                 out_size: int,
                 decomposed_rank: Optional[int] = None,
                 pool_size: Optional[int] = None,
                 ignore_mapping_for_equal_pool: bool = True,
                 use_linear: bool = False,
                 use_bias: bool = False
                 ):
        super().__init__()
        decomposed_rank = decomposed_rank or 1
        pool_size = pool_size or out_size

        self.w_a = nn.Parameter(torch.empty(pool_size, decomposed_rank, left_size))
        self.w_b = nn.Parameter(torch.empty(pool_size, decomposed_rank, right_size))
        nn.init.kaiming_uniform_(self.w_a, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.w_b, nonlinearity='linear')

        if ignore_mapping_for_equal_pool and out_size == pool_size:
            self.w_o = None
        else:
            self.w_o = nn.Parameter(torch.empty(pool_size, out_size))
            nn.init.kaiming_uniform_(self.w_o, nonlinearity='tanh')

        self.linear_a = self.linear_b = None
        if use_linear:
            self.linear_a = nn.Linear(left_size, out_size, bias=False)
            self.linear_b = nn.Linear(right_size, out_size, bias=False)

        self.b = None
        if use_bias:
            self.b = nn.Parameter(torch.zeros(out_size))

    def forward(self, left: torch.Tensor, right: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param left: (*, left)
        :param right: (*, right)
        :return:
        """
        left_size = left.size()
        left = left.reshape(-1, left_size[-1])
        # left_unsqueezed: (-1, 1, 1, left)
        # wa_left: (-1, pool, rank)
        wa_left = (self.w_a * left.unsqueeze(-2).unsqueeze(-2)).sum(dim=-1)

        if right is not None:
            right = right.reshape(-1, right.size()[-1])

            # right_unsqueezed: (-1, 1, 1, right)
            # wb_right: (-1, pool, rank)
            wb_right = (self.w_b * right.unsqueeze(-2).unsqueeze(-2)).sum(dim=-1)

            # lwr: (-1, pool)
            lwr = (wa_left * wb_right).sum(dim=-1)
        else:
            lwr = wa_left.sum(dim=-1)

        if self.w_o is not None:
            # lwr: (-1, out_size)
            lwr = torch.matmul(lwr, self.w_o)

        if self.b is not None:
            lwr = lwr + self.b

        if self.linear_a is not None:
            lwr = lwr + self.linear_a(left)

        if self.linear_b is not None and right is not None:
            lwr = lwr + self.linear_b(right)

        # lwr: (*, out_size)
        return lwr.reshape(*left_size[:-1], -1)




