from typing import Tuple, Dict, List
import torch
from torch import nn

class GrammarTutor(nn.Module):
    def __init__(self, vocab_size, lhs_indexing, padded_rhs_options: torch.Tensor):
        """
        :param vocab_size: int
        :param lhs_indexing: [int] * lhs_num
        :param padded_rhs_options: (lhs_num, max_count, 4, max_seq)
        """
        super().__init__()

        # (vocab, max_count, 4, max_seq)
        g = torch.zeros(vocab_size, *padded_rhs_options.size()[1:], dtype=torch.long)
        for lhs_id, rhs_tensor in zip(lhs_indexing, padded_rhs_options):
            g[lhs_id] = rhs_tensor

        # use the parameter wrapper such that the module will be move to cuda together with the model
        self._g = nn.Parameter(g, requires_grad=False)

    def forward(self, lhs: torch.Tensor) -> torch.LongTensor:
        """
        :param lhs: (batch, *) the LHS id batch
        :return: (batch, *, max_count, 4, max_seq),
                the 4 sequences are symbols, parental_growth, fraternal_growth, and mask, respectively.
        """
        size_pref = lhs.size()

        # motto: (batch, *, max_count, 4, max_seq)
        motto = self._g[lhs]

        # motto_rs: (-1, max_count, 4, max_seq)
        motto_rs = motto.reshape(-1, *motto.size()[-3:])

        # mask: (-1, max_count, max_seq)
        mask = motto_rs[:, :, -1, :]

        # clip the rules if all the companion rules with the same indices w.r.t to any LHS in the batch
        # has been entirely masked (no RHS symbols)
        # clipped_motto: (-1, valid_masks, 4, valid_seq)
        v_clipped_motto = motto_rs[:, :, :, mask.sum([0, 1]) > 0]
        clipped_motto = v_clipped_motto[:, mask.sum([0, 2]) > 0, :, :]

        clipped_motto = clipped_motto.reshape(*size_pref, *clipped_motto.size()[-3:])
        return clipped_motto

