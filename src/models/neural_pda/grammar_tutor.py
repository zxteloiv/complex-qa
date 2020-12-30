from typing import Tuple, Dict, List
import torch
from torch import nn

class GrammarTutorForGeneration(nn.Module):
    def __init__(self, vocab_size, padded_rhs_dict: Dict[int, torch.Tensor]):
        """
        :param vocab_size: int
        :param padded_rhs_dict: {lhs_id: (max_count, 4, max_seq)}
        """
        super().__init__()

        rule_sample = next(iter(padded_rhs_dict.values()))
        # (vocab, max_count, 4, max_seq)
        self._g = nn.Parameter(torch.zeros(vocab_size, *rule_sample.size()).long(), requires_grad=False)
        for lhs_id, rhs_tensor in padded_rhs_dict.items():
            self._g[lhs_id] = rhs_tensor

    def forward(self, lhs: torch.Tensor) -> torch.LongTensor:
        """
        :param lhs: (batch,) the LHS id batch
        :return: (batch, max_count, 4, max_seq),
                the 4 sequences are symbols, parental_growth, fraternal_growth, and mask, respectively.
        """
        # motto: (batch, max_count, 4, max_seq)
        motto = self._g[lhs]

        # mask: (batch, max_count, max_seq)
        mask = motto[:, :, -1, :]

        # clip the rules if all the companion rules with the same indices w.r.t to any LHS in the batch
        # has been entirely masked (no RHS symbols)
        # clipped_motto: (batch, valid_masks, 4, valid_seq)
        v_clipped_motto = motto[:, :, :, mask.sum([0, 1]) > 0]
        clipped_motto = v_clipped_motto[:, mask.sum([0, 2]) > 0, :, :]
        return clipped_motto

