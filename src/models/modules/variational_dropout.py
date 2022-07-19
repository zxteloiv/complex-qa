import torch
from torch import nn


class VariationalDropout(nn.Module):
    """
    A Dropout derived from the Variational Inference perspective.
    The mask is fixed during a time
    """
    def __init__(self, p: float = 0.5, batch_dim: int = 0, vector_dim: int = -1,
                 on_the_fly: bool = True, rescaling: bool = True):
        """

        :param p: the rate to discard
        :param batch_dim: on which dimension is the batch
        :param vector_dim: on which dimension is the hidden states, other dimensions are kept intact
        :param on_the_fly: whether the mask is generated on-the-fly or saved.
                            When set True, the mask will not be saved but the reset is not needed between iterations.
                            When set False, the mask is fixed until reset, useful for iterative usage like the RNN decoder.
        :param rescaling: whether a rescaling is required, it seems unnecessary in AllenNLP but not in FastAI
        """
        super().__init__()

        assert 0 <= p <= 1

        self.p = p
        self._mask = None
        self._batch_dim = batch_dim
        self._vector_dim = vector_dim
        self._on_the_fly = on_the_fly
        self._rescaling = rescaling

    def _init_mask(self, batch_size, vector_size, ndim, device: torch.device):
        size = [1] * ndim
        size[self._batch_dim] = batch_size
        size[self._vector_dim] = vector_size
        return torch.empty(size, device=device).bernoulli_(p=self.p)

    def extra_repr(self) -> str:
        return 'ratio={}, on_the_fly={}, rescaling={}, batch_at_dim={}, vec_at_dim={}'.format(
            self.p, self._on_the_fly, self._rescaling, self._batch_dim, self._vector_dim
        )

    def reset(self):
        self._mask = None

    def forward(self, input):
        if not self.training or self.p == 0:
            return input

        if self._on_the_fly or self._mask is None:
            mask = self._init_mask(input.size()[self._batch_dim], input.size()[self._vector_dim], input.ndim, input.device)
            if not self._on_the_fly:
                self._mask = mask
        else:
            mask = self._mask

        input = input * mask
        if self._rescaling:
            input = input / (1 - self.p)
        return input
