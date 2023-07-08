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
]

T3T = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
T3L = tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
T3F = tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
T4T = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
T4L = tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]
T4F = tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
T5T = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
T5L = tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]
T5F = tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]

Tensor = torch.Tensor
LT = torch.LongTensor
FT = torch.FloatTensor
