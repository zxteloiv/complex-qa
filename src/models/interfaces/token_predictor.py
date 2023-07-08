from typing import Literal
from functools import partialmethod

import torch
from torch import nn
from abc import ABC
from enum import StrEnum, auto


class PredSemantics(StrEnum):
    logits = auto()
    probs = auto()


class TokenPredictor(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._output_sem: str = PredSemantics.logits

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        :param hidden: (*, hidden_size)
        :return: (*, num_toks)
        """
        raise NotImplementedError

    @property
    def output_semantic(self) -> str:
        return self._output_sem

    def output_as(self, value: PredSemantics):
        self._output_sem = value

    output_as_logits = partialmethod(output_as, PredSemantics.logits)
    output_as_probs = partialmethod(output_as, PredSemantics.probs)


