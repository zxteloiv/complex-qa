from typing import Dict, List
import torch
from torch import nn
from torch.nn import functional as F

class ExactTokenTutor(nn.Module):
    def __init__(self, symbol_num, exact_token_num, symbol_token_lookup: Dict[int, List[int]],
                 symbol_unspecified_could_lead_to_any_token: bool = True):
        super().__init__()

        t = torch.zeros(symbol_num, exact_token_num)
        if symbol_unspecified_could_lead_to_any_token:
            torch.fill_(t, 1.)

        for sid, tid_list in symbol_token_lookup.items():
            t[sid] = 0
            for tid in tid_list:
                t[sid, tid] = 1

        self._t = nn.Parameter(t, requires_grad=False)

    def forward(self, symbols):
        """
        :param symbols: (*,) input_symbols
        :return: weights: (*, V) weights for exact tokens conditioned on the input symbol.
        """
        return self._t[symbols]
