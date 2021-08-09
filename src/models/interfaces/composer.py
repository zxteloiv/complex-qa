import torch

class TwoVecComposer(torch.nn.Module):
    def forward(self, val1: torch.Tensor, val2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError
