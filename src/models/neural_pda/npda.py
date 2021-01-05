from typing import Any, Callable, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from allennlp.nn.util import masked_softmax
from ..interfaces.attention import VectorContextComposer
from .batched_stack import BatchStack
from ..interfaces.unified_rnn import UnifiedRNN
from ..modules.stacked_rnn_cell import StackedRNNCell
from utils.seq_collector import SeqCollector
from allennlp.nn.util import min_value_of_dtype, tiny_value_of_dtype
from .tree_state_updater import TreeStateUpdater
from utils.nn import init_state_for_stacked_rnn_with_source

from .tensor_typing_util import *

class NeuralPDA(nn.Module):
    def __init__(self,
                 # modules
                 symbol_embedding: nn.Embedding,

                 grammar_tutor,
                 token_tutor,
                 rhs_expander: StackedRNNCell,
                 stack: BatchStack,

                 tree_state_updater: TreeStateUpdater,

                 symbol_predictor: nn.Module,
                 query_attention_composer: VectorContextComposer,

                 exact_token_predictor: nn.Module,

                 # configuration
                 grammar_entry: int,
                 max_derivation_step: int = 1000,
                 dropout: float = 0.2,
                 ):
        super().__init__()
        self._expander = rhs_expander
        self._stack = stack
        self._gt = grammar_tutor
        self._tt = token_tutor
        self._embedder = symbol_embedding
        self._symbol_predictor = symbol_predictor
        self._exact_token_predictor = exact_token_predictor

        self._query_attn_comp = query_attention_composer
        self._tree_state_udpater: TreeStateUpdater = tree_state_updater
        self._dropout = nn.Dropout(dropout)

        # configurations
        self.grammar_entry = grammar_entry
        self.max_derivation_step = max_derivation_step  # a very large upper limit for the runtime storage

        self.choice_prob_policy = "normalized_logp"

        # the helpful storage for runtime forwarding
        self._query_attn_fn = None
        self._tree_state = None
        self._derivation_step = 0

    def init_automata(self, batch_sz: int, device: torch.device, query_attn_fn: Callable):
        # init stack
        self._stack.reset(batch_sz, device)
        start = torch.full((batch_sz, 1), fill_value=self.grammar_entry, device=device).long()
        self._stack.push(start, push_mask=torch.ones((batch_sz,), device=device, dtype=torch.long))
        # init derivation counter
        self._derivation_step = 0
        # init tree state
        self._tree_state = None
        # since the automata independently runs for every batch, the query attention is also initialized every batch.
        self._query_attn_fn = query_attn_fn

    def continue_derivation(self):
        if self._derivation_step >= self.max_derivation_step:
            return False
        self._derivation_step += 1

        _, success = self._stack.top()
        batch_stack_is_not_empty = success.sum() > 0
        return batch_stack_is_not_empty

    def reset_automata(self):
        # init derivation counter
        self._derivation_step = 0
        # init tree state
        self._tree_state = None
        self._query_attn_fn = None

    def forward(self, token_based=False, **kwargs):
        if not token_based:
            return self._forward_derivation_step(**kwargs)

        else:
            raise NotImplementedError

    def _forward_derivation_step(self, step_symbols=None):
        """
        :param step_symbols: (batch, max_seq), Optional, the gold symbol used,
            the corresponding mask is not given and should be referred to the grammar_guide,
            even though the step_symbol is inconsistent with the grammar, which is hardly possible.

        :return: Tuple of 6 objects
            lhs, lhs_mask: (batch,) ;
            grammar_guide: (batch, opt_num, 4, max_seq) ;
            opts_logp: (batch, opt_num) ;
            topo_preds: Tuple[(batch, max_seq) * 4] ;
            exact_token_p: [(batch, max_seq, V)]
        """
        # lhs, lhs_mask: (batch,)
        # tree_state: (batch, hidden)
        lhs, lhs_mask = self._select_node()
        tree_state = self._encode_partial_tree()

        # grammar_guide: (batch, opt_num, 4, max_seq)
        grammar_guide = self._gt(lhs)

        # opt_prob: (batch, opt_num)
        # rhs_output: (batch, opt_num, hid, seq)
        opt_prob, rhs_output = self._get_topological_choice_distribution(lhs, tree_state, grammar_guide)

        # best_choice: (batch,)
        best_choice = self._make_a_topological_choice(opt_prob, step_symbols, grammar_guide)
        chosen_symbols, chosen_p_growth, chosen_mask = self.retrieve_the_chosen_structure(best_choice, grammar_guide)

        self._update_tree_state(self._retrieve_the_chosen_seq_output(best_choice, rhs_output), chosen_mask)
        exact_token_logit = self._predict_exact_token_after_derivation(chosen_symbols)

        return lhs, lhs_mask, grammar_guide, opt_prob, exact_token_logit

    def _select_node(self) -> Tuple[LT, LT]:
        node, success = self._stack.pop()
        return node.squeeze(-1).long(), success

    def _encode_partial_tree(self):
        return self._tree_state.detach() if self._tree_state is not None else None

    def _get_topological_choice_distribution(self, lhs, tree_state, grammar_guide):
        """
        get the topological choice distribution for the current derivation, conditioning on the given LHS and Tree States
        :param lhs: (batch,)
        :param tree_state: (batch, hid_dim)
        :param grammar_guide: (batch, opt_num, 4, max_seq)
        :return: tuple of 2:
                (batch, opt_num) choice probabilities,
                (batch, opt_num, hid, seq) hidden states of every option route
        """
        mem = SeqCollector()
        valid_symbols = grammar_guide[:, :, 0]  # valid_symbols: (batch, opt_num, max_seq)
        batch, opt_num, max_symbol_len = valid_symbols.size()
        step_inp = lhs.unsqueeze(-1).expand(batch, opt_num)

        for step in range(max_symbol_len):
            if step > 0:    # grammar guide is always available
                step_inp = valid_symbols[:, :, step - 1]

            # Produce a locally-compliant step output and symbol distribution.
            # step_output: (batch, opt_num, hid), to be used to update the tree state
            # symbol_logits: (batch, opt_num, V), to be used to compute the choice log-likelihoods
            step_output, symbol_logits = self._predict_tree_symbol(step_inp, tree_state)
            mem(symbol_logits=symbol_logits, step_output=step_output)

        # rhs_logits: (batch, opt_num, V, seq)
        rhs_logits = mem.get_stacked_tensor('symbol_logits', dim=-1)

        # opt_prob: (batch, opt_num)
        opt_prob = self._gather_and_merge_option_seq(valid_symbols, grammar_guide[:, :, -1], rhs_logits)

        # the hidden output is already fed with valid symbols from every option route
        # rhs_output: (batch, opt_num, hid, seq)
        rhs_output = mem.get_stacked_tensor('step_output', dim=-1)

        return opt_prob, rhs_output

    def _predict_tree_symbol(self, symbol, tree_state) -> Tuple[FT, FT]:
        """
        :param symbol: (batch, opt_num)
        :param tree_state: (batch, hid_sz)
        :return: output (batch, opt_num, hid_sz), step_symbol_logit (batch, opt_num, V)
        """
        emb = self._dropout(self._embedder(symbol)) # (batch, opt_num, emb_sz)

        # the hidden state is discarded each time, hidden inputs of the RNN is constructed from the tree
        # TODO: make sure the expander accepts (batch, opt_num, emb_sz) as input
        hx = None
        if tree_state is not None:
            # use_lowest_for_all
            init_state = init_state_for_stacked_rnn_with_source([tree_state], self._expander.get_layer_num(), "lowest")
            hx, _ = self._expander.init_hidden_states(init_state)
        _, step_out = self._expander(emb, hx)

        # attention computation and compose the context vector with the symbol hidden states
        query_context = self._query_attn_fn(step_out)
        step_out_att = self._query_attn_comp(query_context, step_out)

        # s_logits: (batch, V)
        symbol_logits = self._symbol_predictor(step_out_att)
        return step_out_att, symbol_logits

    def _gather_and_merge_option_seq(self, valid_rhs: LT, rhs_mask: LT, rhs_logits: FT) -> FT:
        """
        :param valid_rhs: (batch, opt_num, max_symbol_len)
        :param rhs_mask: (batch, opt_num, max_symbol_len)
        :param rhs_logits: (batch, opt_num, V, max_symbol_len)
        :return: (batch, opt_num), the probability over the valid derivation options
        """
        # logprob is required s.t. the logits at individual steps are comparable(with prob.) and could be added together(with log)
        # the logit is never manually reset to near -inf, so it's safe to use without an eps.
        rhs_logp = F.log_softmax(rhs_logits, dim=-2)    # (batch, opt_num, V, max_symbol_len)

        # valid_seq_logp: (batch, opt_num, max_symbol_len)
        valid_seq_logp = torch.gather(rhs_logp, dim=2, index=valid_rhs.unsqueeze(2)).squeeze(2)
        # choice_logp: (batch, opt_num)
        choice_logp = (valid_seq_logp * rhs_mask).sum(dim=-1)
        return self._choice_logp_to_choice_distribution(choice_logp)

    def _choice_logp_to_choice_distribution(self, choice_logp: FT) -> FT:
        # choice_logp: (batch, opt_num)
        # the longer the better for the uniform case
        if self.choice_prob_policy == "normalized_logp":
            return choice_logp / (choice_logp.sum(-1, keepdim=True) + 1e-15)

        else:
            raise NotImplementedError

    def _make_a_topological_choice(self, opt_prob, step_symbols: NullOrLT, grammar_guide: NullOrLT) -> LT:
        """
        Choose a route from the options.
        During training the gold option is chosen based on the equality
        between the step_symbols and the option in grammar_guide.
        For evaluation, the greedy search is simply adopted.

        :param opt_prob: (batch, opt_num)
        :param step_symbols: (batch, max_seq)
        :param grammar_guide: (batch, opt_num, 4, seq)
        :return: (batch,) indicating the chosen option id
        """
        if step_symbols is None:    # greedy choice during inference
            best_choice = opt_prob.argmax(dim=-1)

        else:   # find gold choice during training
            best_choice = self.find_choice_by_symbols(step_symbols, grammar_guide)

        return best_choice

    @staticmethod
    def find_choice_by_symbols(step_symbols: LT, grammar_guide: LT) -> LT:
        opt_symbols = grammar_guide[:, :, 0, :]  # (batch, opt_num, max_steps)
        opt_mask = grammar_guide[:, :, -1, :]  # (batch, opt_num, max_steps)

        comp_len = grammar_guide.size()[-1]
        gold_symbols = step_symbols[:, :comp_len]

        # seq_comp: (batch, opt_num, compatible_len)
        seq_comp = (opt_symbols == gold_symbols.unsqueeze(1)) * opt_mask
        _choice = (seq_comp.sum(dim=-1) == opt_mask.sum(dim=-1))  # _choice: (batch, opt_num)
        # there must be some choice found for all batch, otherwise the data is inconsistent
        assert (_choice.sum(dim=1) > 0).all()

        # argmax is forbidden on bool storage, but max is ok
        return _choice.max(dim=-1)[1]

    def _update_tree_state(self, rhs_output, rhs_mask) -> None:
        """
        :param rhs_output: (batch, hid, seq)
        :param rhs_mask: (batch, seq)
        :return:
        """
        # tree_state: (batch, hid)
        tree_state = self._tree_state
        self._tree_state = self._tree_state_udpater(tree_state, rhs_output, rhs_mask).detach()

    def _predict_exact_token_after_derivation(self, symbols: LT) -> FT:
        # symbols: (batch, expansion_len)
        expansion_len = symbols.size()[-1]

        # compliant_weights: (batch, seqlen, V)
        compliant_weights = self._tt(symbols)

        # symbol_emb: (batch, *, emb)
        symbol_emb = self._dropout(self._embedder(symbols))
        # tree_state: (batch, expansion_len, hid) <- (batch, 1, hid) <- (batch, hid)
        tree_state = self._encode_partial_tree().unsqueeze(1).expand(-1, expansion_len, -1)
        proj_inp = torch.cat([symbol_emb, tree_state], dim=-1)
        # exact_logit: (batch, *, V)
        exact_logit = self._exact_token_predictor(proj_inp)
        exact_logit = exact_logit + (compliant_weights + tiny_value_of_dtype(exact_logit.dtype)).log()
        return exact_logit

    def push_predictions_onto_stack(self, symbol: LT, p_growth: LT, symbol_mask: LT, lhs_mask: NullOrLT):
        """
        Iterate the symbols backwards and push them onto the stack based on topological information.

        :param symbol: (batch, max_seq)
        :param p_growth: (batch, max_seq), 1 if the symbol is a non-terminal and will grow a subtree, 0 otherwise.
        :param symbol_mask: (batch, max_seq)
        :param lhs_mask: (batch,), will be absent if the symbols are gold, and otherwise must be provided.
        :return:
        """
        for step in reversed(range(symbol.size()[-1])):
            push_mask = p_growth[:, step] * symbol_mask[:, step]
            if lhs_mask is not None:
                push_mask *= lhs_mask
            assert push_mask.ndim == 1
            self._stack.push(symbol[:, step].unsqueeze(-1), push_mask.long())

    @staticmethod
    def retrieve_the_chosen_structure(choice, grammar_guide) -> T3L:
        """
        Inference the tree topological structure of a derivation.
        Exact token inference is not considered here.
        :param choice: (batch,)
        :param grammar_guide: (batch, opt_num, 4, max_seq)
        :return: Tuple[(batch, seq) * 4], symbol sequence, parental growth, fraternal growth, and mask respectively.
        """
        batch_index = torch.arange(choice.size()[0], device=choice.device)

        # topo_pred: (batch, 4, max_seq)
        topology_predictions = grammar_guide[batch_index, choice]

        symbol = topology_predictions[:, 0, :]
        p_growth = topology_predictions[:, 1, :]
        symbol_mask = topology_predictions[:, -1, :]

        return symbol, p_growth, symbol_mask

    @staticmethod
    def _retrieve_the_chosen_seq_output(choice, rhs_output):
        """
        :param choice: (batch, )
        :param rhs_output: (batch, opt_num, hid, seq_len)
        :return:
        """
        batch_index = torch.arange(choice.size()[0], device=choice.device)
        return rhs_output[batch_index, choice]
