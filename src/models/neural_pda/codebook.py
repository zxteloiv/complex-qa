from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

class CodeBook(nn.Module):
    def __init__(self,
                 num_nonterminals: int,
                 nonterminal_dim: int,
                 init_codebook_confidence: int = 1,  # the number of iterations from which the codebook is learnt
                 codebook_training_decay: float = 0.99,
                 ):
        super().__init__()
        self.num_nt = num_nonterminals
        self.nonterminal_dim = nonterminal_dim

        codebook = F.normalize(torch.randn(num_nonterminals, nonterminal_dim)).detach()
        self.codebook = nn.Parameter(codebook)
        self._code_m_acc = nn.Parameter(codebook.clone())
        self._code_n_acc = nn.Parameter(torch.full((num_nonterminals,), init_codebook_confidence))
        self._code_decay = codebook_training_decay

    def quantize_code(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param codes: (batch_size, -1, hidden_dim)
        :return: a tuple of
                (batch_size, n, hidden_dim), the quantized code vector, selected directly from the codebook
                (batch_size, n), the id of the code in the codebook
        """
        # codes: (B, n, hidden_dim)
        old_shape = codes.size()
        hidden_dim = old_shape[-1]

        # code_rs: (..., 1, hidden_dim)
        # self.codebook: (#NT, hidden_dim) -> (1, #NT, hidden_dim)
        # diff_vec: (..., #NT, hidden_dim)
        code_rs = codes.reshape(-1, 1, hidden_dim)
        diff_vec = code_rs - torch.unsqueeze(self.codebook, dim=0)

        # diff_norm: (..., #NT)
        # quantized_idx: (...,)
        diff_norm = torch.norm(diff_vec, dim=2)
        quantized_idx = torch.argmin(diff_norm, dim=1)

        quantized_codes = self.codebook[quantized_idx]
        return quantized_codes.reshape(*old_shape), quantized_idx.reshape(*old_shape[:-1])

    forward = quantize_code

    def get_step_moving_averages(self,
                                  codes: torch.Tensor,
                                  quantized_idx: torch.Tensor,
                                  updating_mask: torch.Tensor):
        """
        Compute and return the moving average accumulators (m and n).
        :param codes: (batch, 2, hidden_dim)
        :param quantized_idx: (batch, 2)
        :param updating_mask: (batch,)
        :return: (#NT, d), (#NT) the accumulator of each timestep for the nominator and denominators.
        """
        # qidx_onehot, filtered_qidx: (batch, 2, #NT)
        qidx_onehot = (quantized_idx.unsqueeze(-1) == torch.arange(self.num_nt, device=codes.device).reshape(1, 1, -1))
        filtered_qidx = updating_mask.reshape(-1, 1, 1) * qidx_onehot

        # n_t: (#NT,) the number of current codes quantized for each NT
        # m_t: (#NT, d)
        n_t = filtered_qidx.sum(0).sum(0)
        m_t = (filtered_qidx.unsqueeze(-1) * codes.unsqueeze(-2)).sum(0).sum(0)

        return m_t, n_t

    def update_codebook(self):
        """Called when the codebook parameters need update, after the optimizer update step perhaps"""
        if self.training:
            self.codebook = nn.Parameter(self._code_m_acc / (self._code_n_acc.unsqueeze(-1) + 1e-15))

    def update_accumulator(self, acc_m, acc_n):
        if self.training:
            r = self._code_decay
            self._code_m_acc = nn.Parameter(r * self._code_m_acc + (1 - r) * acc_m)
            self._code_n_acc = nn.Parameter(r * self._code_n_acc + (1 - r) * acc_n)




