from abc import ABC

from torch import nn
import torch
from .tensor_typing_util import *
from ..modules.container import MultiInputsSequential

class RuleScorer(nn.Module, ABC):
    def forward(self, rule_option: FT, query_context: FT, tree_state: FT) -> FT:
        """
        Compute the distribution of the rule options
        by the given rule representation, query attentive context of the rules,
        and the current tree state.

        :param rule_option: (batch, opt_num, hid)
        :param query_context: (batch, opt_num, context_sz)
        :param tree_state: (batch, hid)
        :return: the logits over the rule options.
        """
        raise NotImplementedError

class MLPScorerWrapper(RuleScorer):
    def __init__(self, module: MultiInputsSequential):
        super().__init__()
        self._module = module

    def forward(self, rule_option: FT, query_context: FT, tree_state: NullOrFT) -> FT:
        # transform tree state to the appropriate size
        # and squeeze the linear mapping of the module, which usually has a top layer with output dim=1 for logits
        if tree_state is not None:
            ts = tree_state.unsqueeze(1).expand_as(rule_option)
        else:
            ts = torch.zeros_like(rule_option)

        inp = torch.cat([rule_option, query_context, ts], dim=-1)
        return self._module(inp).squeeze(-1)
