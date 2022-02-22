from abc import ABC
from typing import List, Optional, Callable, Any, Tuple, Union, Sequence
import torch


T_HIDDEN = Union[None, torch.Tensor, Sequence[torch.Tensor]]


class _RNNBase(torch.nn.Module, ABC):

    def get_output_state(self, hidden: Any) -> torch.Tensor:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class UnifiedRNN(_RNNBase, ABC):

    def forward(self, inputs, hidden: Any) -> Tuple[Any, torch.Tensor]:
        # in a pure RNN cell, hidden is something about its internal
        # in a stacked rnn however, hidden is a list of hidden states of each layer
        # they are all unknown structures and thus typed by Any
        raise NotImplementedError

    def init_hidden_states(self, forward_out: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """Initialize the hidden states with an unknown internal structure from the forward_out tensor"""
        raise NotImplementedError


class RNNStack(_RNNBase, ABC):
    def forward(self, inputs, hidden: List[Any]) -> Tuple[Any, torch.Tensor]:
        # at least the hidden is a list indicating layers
        raise NotImplementedError

    def get_layer_num(self) -> int:
        raise NotImplementedError

    def init_hidden_states(self, tensor_list: List[torch.Tensor]) -> Tuple[Any, torch.Tensor]:
        raise NotImplementedError

    def get_layered_output_state(self, hidden: List[Any]) -> List[torch.Tensor]:
        raise NotImplementedError


class _RNNEncoderBase(torch.nn.Module, ABC):

    def is_bidirectional(self) -> bool:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class EncoderRNN(_RNNEncoderBase, ABC):
    def forward(self, inputs, mask, hidden) -> torch.Tensor:
        """returns the output at all timesteps"""
        raise NotImplementedError


class EncoderRNNStack(_RNNEncoderBase, ABC):
    def get_layer_num(self) -> int:
        raise NotImplementedError

    def forward(self, inputs, mask, hidden) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """returns the last encoder"""
        raise NotImplementedError

    def get_last_layered_output(self) -> List[torch.Tensor]:
        """return the output of every layer"""
        raise NotImplementedError


class HiddenAwareRNN(torch.nn.Module):
    @staticmethod
    def weighted_sum_single_var(var_list, weight):
        # var_list: [var]
        # var: (batch, hidden_emb)
        # stacked: (batch, hidden_emb, total)
        stacked = torch.stack(var_list, dim=-1)

        # weight: (batch, 1, total) <- (batch, total)
        weight = weight.unsqueeze(1)

        merged = (stacked * weight).sum(2)
        return merged

    def merge_hidden_list(self, hidden_list, weight) -> Any:
        """
        Merge the hidden_list using weighted sum.
        Hidden States can not be directly processed therefore they are only possible contained in a list.

        Hidden States internals must be kept unknown to the outside,
        so merging the hidden list is also rarely used.
        You can choose not to implement it in your derived RNN, unless the model explicitly requires.

        :param hidden_list: [hidden] or [(hidden, context)] or else
        :param weight: (batch, total = len(hidden_list) )
        :return: hidden or (hidden, context), or something else if you know about its internals
        """
        # This method is seldom required thus we provide an implementation directly in the interface.
        # raise NotImplementedError
        return self.weighted_sum_single_var(hidden_list, weight)

