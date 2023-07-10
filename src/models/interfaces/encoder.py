from abc import ABC

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
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """returns the output at all timesteps"""
        raise NotImplementedError


class StackEncoder(Encoder, ABC):
    """
    An encoder stack accepts an embedding tensor and output another tensor,
    but stores the intermediate outputs for possible future uses.
    """
    def get_layer_num(self) -> int:
        """the number of stacked layers"""
        raise NotImplementedError

    def __len__(self):
        return self.get_layer_num()

    def get_layered_output(self) -> list[torch.Tensor]:
        """return the output of every intermediate layer after the last forward"""
        raise NotImplementedError


class EmbedAndEncode(torch.nn.Module, ABC):
    """
    A basic interface assumed to be part of the seq2seq framework.
    Get hidden states direct from the tensor of token ids, which is the key point.

    A typical implementation may include an nn.Embedding followed by an EncoderStack,
    but another popular choice would be a wrapper of BERT models.
    """
    def forward(self, tokens: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """accepts the token id sequence, returns layered states and state masks"""
        raise NotImplementedError

    def is_bidirectional(self):
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class EmbedAndGraphEncode(EmbedAndEncode, ABC):
    def set_graph(self, graph: torch.Tensor):
        raise NotImplementedError
