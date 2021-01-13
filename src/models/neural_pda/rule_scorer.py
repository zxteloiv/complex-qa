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
        # the tree state and query context should have similar size prefix as the rule_option,
        # except for the option dimension.
        opt_num = rule_option.size()[-2]

        expand_fn = lambda t: t.unsqueeze(-2).expand(*t.size()[:-1], opt_num, t.size()[-1])
        tree_state, query_context = list(map(expand_fn, (tree_state, query_context)))

        inp = torch.cat([rule_option, query_context, tree_state], dim=-1)
        return self._module(inp).squeeze(-1)
