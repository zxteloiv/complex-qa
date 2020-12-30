from typing import Tuple, Optional, Union
import torch

__all__ = [
    "T3T",
    "T3L",
    "T3F",
    "T4T",
    "T4L",
    "T4F",
    "T5T",
    "T5L",
    "T5F",
    "Tensor",
    "LT",
    "FT",
    "Nullable",
    "NullOrT",
    "NullOrFT",
    "NullOrLT",
]

T3T = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
T3L = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
T3F = Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
T4T = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
T4L = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]
T4F = Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
T5T = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
T5L = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]
T5F = Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]

Tensor = torch.Tensor
LT = torch.LongTensor
FT = torch.FloatTensor

Nullable = Optional

NullOrT = Union[None, Tensor]
NullOrLT = Union[None, LT]
NullOrFT = Union[None, FT]
