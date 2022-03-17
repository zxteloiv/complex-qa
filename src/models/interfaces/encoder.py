from abc import ABC
from typing import Union, Tuple, List

import torch


class _EncoderBase(torch.nn.Module, ABC):

    def is_bidirectional(self) -> bool:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class Encoder(_EncoderBase, ABC):
    """
    An encoder accepts an embedding tensor and outputs another tensor
    """
    def forward(self, inputs, mask, hidden) -> torch.Tensor:
        """returns the output at all timesteps"""
        raise NotImplementedError


class EncoderStack(_EncoderBase, ABC):
    """
    An encoder stack accepts an embedding tensor and output another tensor,
    but stores the intermediate outputs for possible future uses.
    """
    def get_layer_num(self) -> int:
        """the number of stacked layers"""
        raise NotImplementedError

    def __len__(self):
        return self.get_layer_num()

    def forward(self, inputs, mask) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """returns the last encoder output, initialized from zero, without hidden"""
        raise NotImplementedError

    def get_layered_output(self) -> List[torch.Tensor]:
        """return the output of every layer after the last forward"""
        raise NotImplementedError