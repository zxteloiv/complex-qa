import torch
from torch import nn

class VariationalDropout(nn.Module):
    """
    A Dropout derived from the Variational Inference perspective.
    The mask is fixed during a time
    """
    def __init__(self, p: float = 0.5, batch_dim: int = 0, vector_dim: int = -1):
        super().__init__()

        assert 0 < p < 1

        self.p = p
        self._mask = None
        self._batch_dim = batch_dim
        self._vector_dim = vector_dim

    def init_mask(self, batch_size, vector_size, ndim, device: torch.device):
        size = [1] * ndim
        size[self._batch_dim] = batch_size
        size[self._vector_dim] = vector_size
        self._mask = torch.empty(size, device=device).bernoulli_(p=self.p)

    def reset(self):
        self._mask = None

    def forward(self, input):
        if not self.training:
            return input

        if self._mask is None:
            self.init_mask(input.size()[self._batch_dim], input.size()[self._vector_dim], input.ndim, input.device)

        # if all neurons are masked out the mask shall not be applied
        if (self._mask == 0).all():
            return input

        input = (input * self._mask) / (1 - self.p)
        return input


