from typing import List, Optional, Callable, Any, Tuple
import torch
from utils.nn import filter_cat

class UnifiedRNN(torch.nn.Module):

    def forward(self, inputs, hidden, input_aux: Optional[List] = None) -> Tuple[Any, torch.Tensor]:
        if input_aux is not None:
            inputs = filter_cat([inputs] + input_aux, dim=-1)

        return self._forward_internal(inputs, hidden)

    def _forward_internal(self, inputs, hidden) -> Tuple[Any, torch.Tensor]:
        raise NotImplementedError

    def get_output_state(self, hidden) -> torch.Tensor:
        raise NotImplementedError

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
        raise NotImplementedError

    def init_hidden_states(self, forward_out) -> Tuple[Any, torch.Tensor]:
        raise NotImplementedError
