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
        :param tree_state: (batch, opt_num, hid)
        :return: the logits over the rule options.
        """
        raise NotImplementedError

class MLPScorerWrapper(RuleScorer):
    def __init__(self, module: MultiInputsSequential):
        super().__init__()
        self._module = module

    def forward(self, rule_option: FT, query_context: FT, tree_state: FT) -> FT:
        inp = torch.cat([rule_option, query_context, tree_state], dim=-1)
        return self._module(inp).squeeze(-1)

class HeuristicMLPScorerWrapper(RuleScorer):
    def __init__(self, module: MultiInputsSequential):
        super().__init__()
        self._module = module

    def forward(self, rule_option: FT, query_context: FT, tree_state: FT) -> FT:
        inp = torch.cat([rule_option, query_context, tree_state, query_context - tree_state, query_context * tree_state], dim=-1)
        return self._module(inp).squeeze(-1)

class GeneralizedInnerProductScorer(RuleScorer):
    def __init__(self, normalized: bool = False):
        super().__init__()
        self.normalized = normalized

    def forward(self, rule_option: FT, query_context: FT, tree_state: FT) -> FT:
        """
        Use dot product for all
        :param rule_option: (batch, opt_num, hid)
        :param query_context: (batch, opt_num, hid)
        :param tree_state: (batch, opt_num, hid)
        :return: the logits over the rule options. (batch, opt_num)
        """
        inp = (rule_option * query_context * tree_state).sum(dim=-1)
        if self.normalized:
            inp = inp / (rule_option.norm(dim=-1) + 1e-15)
            inp = inp / (query_context.norm(dim=-1) + 1e-15)
            inp = inp / (tree_state.norm(dim=-1) + 1e-15)

        return inp

class ConcatInnerProductScorer(RuleScorer):
    def __init__(self, module: MultiInputsSequential):
        super().__init__()
        self._module = module

    def forward(self, rule_option: FT, query_context: FT, tree_state: FT) -> FT:
        """
        Use dot product for all
        :param rule_option: (batch, opt_num, hid)
        :param query_context: (batch, opt_num, hid)
        :param tree_state: (batch, opt_num, hid)
        :return: the logits over the rule options. (batch, opt_num)
        """

        # state: (batch, opt_num, hid)
        state = self._module(torch.cat([query_context, tree_state], dim=-1))
        inp = (rule_option * state).sum(dim=-1)
        return inp

class AddInnerProductScorer(RuleScorer):
    def __init__(self, hidden_sz: int, use_layer_norm: bool = True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_sz) if use_layer_norm else None

    def forward(self, rule_option: FT, query_context: FT, tree_state: FT) -> FT:
        """
        Use dot product for all
        :param rule_option: (batch, opt_num, hid)
        :param query_context: (batch, opt_num, hid)
        :param tree_state: (batch, opt_num, hid)
        :return: the logits over the rule options. (batch, opt_num)
        """

        # state: (batch, opt_num, hid)
        state = query_context + tree_state
        if self.layer_norm is not None:
            state = self.layer_norm(state)
        inp = (rule_option * state).sum(dim=-1)
        return inp
