from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

class Codebook(nn.Module):
    def quantize_code(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, codes):
        return self.quantize_code(codes)

    def get_step_moving_averages(self, codes: torch.Tensor, quantized_idx: torch.Tensor, updating_mask: torch.Tensor):
        raise NotImplementedError

    def update_codebook(self):
        raise NotImplementedError

    def update_accumulator(self, acc_m, acc_n):
        raise NotImplementedError

    def get_code_by_id(self, quant_idx: torch.Tensor):
        raise NotImplementedError

    @staticmethod
    def quantize(codebook: torch.Tensor, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param codebook: (#NT, hidden_dim)
        :param codes: (..., hidden_dim)
        :return: a tuple of
                (..., hidden_dim), the quantized code vector, selected directly from the codebook
                (...), the id of the code in the codebook
        """
        # code_rs: (..., 1, hidden_dim)
        code_rs = codes.unsqueeze(-2)

        # dist: (..., #NT)
        dist = (code_rs - codebook).norm(dim=-1)

        # quant_id: (...,)
        # quant_codes: (..., hidden_dim)
        quant_id = torch.argmin(dist, dim=-1)
        quant_codes = codebook[quant_id]

        return quant_codes, quant_id


class VanillaCodebook(Codebook):
    def __init__(self,
                 num_nonterminals: int,
                 nonterminal_dim: int,
                 init_codebook_confidence: float = 1.,  # the number of iterations from which the codebook is learnt
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
        return Codebook.quantize(self.codebook, codes)

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
        qidx_onehot = (quantized_idx.unsqueeze(-1) == torch.arange(self.num_nt, device=codes.device))
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

    def get_code_by_id(self, quant_idx: torch.Tensor):
        return self.codebook[quant_idx]


class SplitCodebook(Codebook):
    def __init__(self,
                 num_nonterminals: int,
                 nonterminal_dim: int,
                 num_splits: int,
                 init_codebook_confidence: float = 1.,  # the number of iterations from which the codebook is learnt
                 codebook_training_decay: float = 0.99,
                 ):
        super().__init__()
        self.num_nt = num_nonterminals  # K
        self.num_splits = num_splits    # N
        self.nonterminal_dim = nonterminal_dim  # d

        self._split_dim = nonterminal_dim // num_splits  # d / N
        self._split_num_nt = round(num_nonterminals ** (1 / num_splits))

        # qid_base := split_nt ** [num_splits]
        self._qid_base = [self._split_num_nt ** k for k in range(num_splits)]

        codebook = F.normalize(torch.randn(self._split_num_nt, self._split_dim)).detach()
        self.codebook = nn.Parameter(codebook)
        self._code_m_acc = nn.Parameter(codebook.clone())
        self._code_n_acc = nn.Parameter(torch.full((self._split_num_nt,), init_codebook_confidence))
        self._code_decay = codebook_training_decay

    def quantize_code(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param codes: (..., hidden_dim)
        :return: a tuple of
                (..., hidden_dim), the quantized code vector, selected directly from the codebook
                (...), the id of the code in the codebook
        """

        # split_codes: (..., num_split, hidden // num_split)
        split_codes = self._get_split_quant_code(codes)

        # split_quant_codes: (..., num_split, split_dim)
        # split_quant_idx: (..., num_split)
        split_quant_codes, split_quant_idx = Codebook.quantize(self.codebook, split_codes)

        # quant_codes: (..., hidden_dim)
        quant_codes = split_quant_codes.reshape(*codes.size())

        # dot_prod at the last dimension
        # (..., num_split)
        quant_idx = (split_quant_idx * torch.tensor(self._qid_base, dtype=torch.int, device=codes.device)).sum(-1)

        return quant_codes, quant_idx

    def get_step_moving_averages(self,
                                 codes: torch.Tensor,
                                 quantized_idx: torch.Tensor,
                                 updating_mask: torch.Tensor):
        """
        Compute and return the moving average accumulators (m and n).
        :param codes: (batch, ..., hidden_dim)
        :param quantized_idx: (batch, ...,)
        :param updating_mask: (batch,)
        :return: (#NT, d), (#NT) the accumulator of each timestep for the nominator and denominators.
        """
        # mod and reminder for the base conversion
        # (batch, ..., num_splits)
        split_quant_idx = self._get_split_quant_idx(quantized_idx)

        # split_onehot_qidx: (batch, ..., num_splits, #split_nt)
        split_id_seq = torch.arange(self._split_num_nt, device=split_quant_idx.device)
        split_onehot_qidx = (split_quant_idx.unsqueeze(-1) == split_id_seq)

        # valid_qidx: (batch, ..., num_splits, #split_nt)
        valid_qidx = updating_mask.reshape(-1, *(1 for _ in range(split_onehot_qidx.ndim - 1))) * split_onehot_qidx

        # n_t: (#split_nt,)
        n_t = valid_qidx.sum_to_size(self._split_num_nt)

        # split_codes: (..., num_split, split_dim)
        split_codes = self._get_split_quant_code(codes)
        # m_t: (#split_nt, split_dim)
        m_t = (valid_qidx.unsqueeze(-1) * split_codes.unsqueeze(-2)
               ).sum_to_size(self._split_num_nt, self._split_dim)

        return m_t, n_t

    def get_code_by_id(self, quant_idx: torch.Tensor):
        # quant_idx: (...,)
        # split_qidx: (..., num_splits)
        # split_quant_codes: (..., num_splits, split_dim)
        split_qidx = self._get_split_quant_idx(quant_idx)
        split_quant_codes = self.codebook[split_qidx]

        # quant_codes: (..., num_splits * split_dim)
        quant_codes = split_quant_codes.reshape(*split_qidx.size()[:-1], -1)
        return quant_codes

    update_codebook = VanillaCodebook.update_codebook
    update_accumulator = VanillaCodebook.update_accumulator

    def _get_split_quant_idx(self, quant_idx):
        reminder = quant_idx
        split_quant_idx = []
        for base in range(self.num_splits - 1, -1, -1):
            # (batch, ...,)
            quotient = reminder // self._qid_base[base]
            reminder = reminder % self._qid_base[base]
            split_quant_idx.append(quotient)
        # (batch, ..., num_splits)
        split_quant_idx.reverse()
        split_quant_idx = torch.stack(split_quant_idx, dim=-1)
        return split_quant_idx

    def _get_split_quant_code(self, codes):
        # split_codes: (..., num_split, hidden // num_split)
        split_codes = torch.stack(codes.split(self._split_dim, dim=-1), dim=-2)
        return split_codes

if __name__ == '__main__':
    codebook = SplitCodebook(16, 10, 2)
    codebook.train()
    batch, hidden_dim = 5, 10
    codes = torch.randn(batch, 2, hidden_dim)
    quant_codes, quant_idx = codebook.quantize_code(codes)
    assert quant_codes.size() == (batch, 2, hidden_dim)
    assert quant_idx.size() == (batch, 2)

    retrieved_qcodes = codebook.get_code_by_id(quant_idx)

    assert (retrieved_qcodes == quant_codes).all()

    m, n = codebook.get_step_moving_averages(codes, quant_idx, torch.ones((batch,)))

    codebook.update_accumulator(m, n)
    codebook.update_codebook()









