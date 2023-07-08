from abc import ABC
from collections.abc import Sequence
import torch


# at least we assume the hidden states are a single tensor or a sequence of tensors.
# by design the user of RNNs such as the Seq2Seq model must not be aware of the internal structures of hidden states.
# each hidden variable within a T_HIDDEN instance represents the hx of an entire batch.
T_HIDDEN = torch.Tensor | Sequence[torch.Tensor]


class _RNNBase(torch.nn.Module, ABC):

    def get_output_state(self, hidden: T_HIDDEN) -> torch.Tensor:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class UnifiedRNN(_RNNBase, ABC):

    def forward(self, inputs: torch.Tensor, hidden: T_HIDDEN | None) -> tuple[T_HIDDEN, torch.Tensor]:
        # in a pure RNN cell, hidden is something about its internal
        # in a stacked rnn however, hidden is a list of hidden states of each layer
        raise NotImplementedError

    def init_hidden_states(self, forward_out: torch.Tensor) -> T_HIDDEN:
        """Initialize the hidden states with an unknown internal structure from the forward_out tensor"""
        raise NotImplementedError


class RNNStack(_RNNBase, ABC):
    def forward(self, inputs: torch.Tensor, hidden: list[T_HIDDEN]) -> tuple[T_HIDDEN, torch.Tensor]:
        # at least the hidden is a list indicating layers
        raise NotImplementedError

    def get_layer_num(self) -> int:
        raise NotImplementedError

    def init_hidden_states(self, tensor_list: list[torch.Tensor]) -> list[T_HIDDEN]:
        raise NotImplementedError

    def get_layered_output_state(self, hidden: list[T_HIDDEN]) -> list[torch.Tensor]:
        raise NotImplementedError


class HiddenAwareRNN(torch.nn.Module):
    @staticmethod
    def weighted_sum_single_var(var_list: list[torch.Tensor], weight: torch.Tensor) -> torch.Tensor:
        # var_list: [var]
        # weight: (batch, total)
        # var: (batch, hidden_emb)
        # stacked: (batch, hidden_emb, total)
        stacked = torch.stack(var_list, dim=-1)
        merged = (stacked * weight[:, None, :]).sum(2)  # (batch, hidden_emb)
        return merged

    def merge_hidden_list(self, hidden_list: list[T_HIDDEN], weight: torch.Tensor) -> T_HIDDEN:
        """
        Merge the hidden_list using weighted sum.
        Hidden States can not be directly processed therefore they are only possible contained in a list.

        Hidden States internals must be kept unknown to the outside,
        so merging the hidden list is also rarely used.
        You can choose not to implement it in your derived RNN, unless the model explicitly requires.

        :param hidden_list: [T_HIDDEN], each T_HIDDEN
        :param weight: (batch, total = len(hidden_list) )
        :return: hidden or (hidden, context), or something else if you know about its internals
        """
        # This method is seldom required thus we provide an implementation directly in the interface.
        raise NotImplementedError

