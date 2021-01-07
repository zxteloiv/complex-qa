from typing import Any, Callable, Optional, Tuple, Literal
import torch
from torch import nn
from torch.nn import functional as F

from allennlp.nn.util import masked_softmax
from ..interfaces.attention import VectorContextComposer
from .batched_stack import BatchStack, TensorBatchStack
from ..interfaces.unified_rnn import UnifiedRNN
from ..modules.stacked_rnn_cell import StackedRNNCell
from utils.seq_collector import SeqCollector
from allennlp.nn.util import min_value_of_dtype, tiny_value_of_dtype
from .tree_state_updater import TreeStateUpdater
from utils.nn import init_state_for_stacked_rnn, get_final_encoder_states

from .tensor_typing_util import *

class NeuralPDA(nn.Module):
    def __init__(self,
                 # modules
                 symbol_embedding: nn.Embedding,

                 grammar_tutor,
                 token_tutor,
                 rhs_expander: StackedRNNCell,
                 stack: TensorBatchStack,

                 tree_state_updater: TreeStateUpdater,
                 stack_node_composer,

                 query_attention_composer: VectorContextComposer,
                 rule_representation_scorer,

                 exact_token_predictor: nn.Module,

                 # configuration
                 grammar_entry: int,
                 max_derivation_step: int = 1000,
                 dropout: float = 0.2,
                 tree_state_policy: Literal["pre_expansion", "post_expansion"] = "post_expansion",
                 ):
        super().__init__()
        self._expander = rhs_expander
        self._stack = stack
        self._gt = grammar_tutor
        self._tt = token_tutor
        self._embedder = symbol_embedding
        self._exact_token_predictor = exact_token_predictor

        self._query_attn_comp = query_attention_composer
        self._tree_state_udpater: TreeStateUpdater = tree_state_updater
        self._dropout = nn.Dropout(dropout)

        self._stack_node_composer = stack_node_composer
        self._rule_scorer = rule_representation_scorer

        # configurations
        self.grammar_entry = grammar_entry
        self.max_derivation_step = max_derivation_step  # a very large upper limit for the runtime storage

        self.tree_state_policy = tree_state_policy

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
        # rhs_output: (batch, opt_num, hid)
        opt_prob, opt_repr = self._get_topological_choice_distribution(lhs, tree_state, grammar_guide)

        # best_choice: (batch,)
        best_choice = self._make_a_topological_choice(opt_prob, step_symbols, grammar_guide)
        chosen_symbols, _, _ = self.retrieve_the_chosen_structure(best_choice, grammar_guide)

        self._update_tree_state(self._retrieve_the_selected_option_repr(best_choice, opt_repr))
        exact_token_logit = self._predict_exact_token_after_derivation(chosen_symbols)

        return lhs, lhs_mask, grammar_guide, opt_prob, exact_token_logit

    def _select_node(self) -> Tuple[LT, LT]:
        node, success = self._stack.pop()
        return node.squeeze(-1).long(), success

    def _encode_partial_tree(self):
        if self.tree_state_policy == "pre_expansion":
            symbols, symbol_mask = self._stack.dump()   # mask: (batch, max_cur)
            symbols = symbols.squeeze(-1)               # symbols: (batch, max_cur)
            symbol_emb = self._embedder(symbols)        # symbol_emb: (batch, max_cur, emb_sz)
            return self._stack_node_composer(symbol_emb, symbol_mask)

        else:
            return self._tree_state.detach() if self._tree_state is not None else None

    def _get_topological_choice_distribution(self, lhs, tree_state, grammar_guide):
        """
        get the topological choice distribution for the current derivation, conditioning on the given LHS and Tree States
        :param lhs: (batch,)
        :param tree_state: (batch, hid_dim)
        :param grammar_guide: (batch, opt_num, 4, max_seq)
        :return: tuple of 2:
                (batch, opt_num) choice probabilities,
                (batch, opt_num, hid) compositional representation of each rule option
        """
        mem = SeqCollector()
        valid_symbols = grammar_guide[:, :, 0]      # valid_symbols: (batch, opt_num, max_seq)
        parental_growth = grammar_guide[:, :, 1]    # p_growth: (batch, opt_num, max_seq)
        fraternal_growth = grammar_guide[:, :, 2]   # f_growth: (batch, opt_num, max_seq)
        batch, opt_num, max_symbol_len = valid_symbols.size()

        hx = None
        if tree_state is not None:
            init_state = init_state_for_stacked_rnn([tree_state], self._expander.get_layer_num(), "all")
            hx, _ = self._expander.init_hidden_states(init_state)

        for step in range(-1, max_symbol_len):
            if step < 0:
                step_symbol = lhs.unsqueeze(-1).expand(batch, opt_num)
                step_p_growth = torch.ones_like(step_symbol).unsqueeze(-1)
                step_f_growth = step_p_growth
            else:
                # grammar guide is always available
                step_symbol = valid_symbols[:, :, step]
                step_p_growth = parental_growth[:, :, step].unsqueeze(-1)
                step_f_growth = fraternal_growth[:, :, step].unsqueeze(-1)

            emb = self._dropout(self._embedder(step_symbol))  # (batch, opt_num, emb_sz)
            cell_input = torch.cat([emb, step_p_growth, step_f_growth], dim=-1)
            # step_output: (batch, opt_num, hid)
            hx, step_out = self._expander(cell_input, hx)

            if step >= 0:
                # attention computation and compose the context vector with the symbol hidden states
                query_context = self._query_attn_fn(step_out)
                # step_output_att: (batch, opt_num, hid)
                step_out_att = self._query_attn_comp(query_context, step_out)
                mem(step_output=step_out_att)

        # rhs_output: (batch, opt_num, max_seq, hid)
        rhs_output = mem.get_stacked_tensor('step_output', dim=-2)
        # symbol_mask: (batch, opt_num, max_seq)
        symbol_mask = grammar_guide[:, :, 3]

        # opt_repr: (batch, opt_num, hid)
        opt_repr = get_final_encoder_states(
            rhs_output.reshape(batch * opt_num, max_symbol_len, -1),
            symbol_mask.reshape(batch * opt_num, max_symbol_len)
        ).reshape(batch, opt_num, -1)

        # option rule score
        scores = self._rule_scorer(opt_repr)    # (batch, opt_num, 1)
        opt_prob = F.softmax(scores.squeeze(-1), dim=-1)    # (batch, opt_num)
        return opt_prob, opt_repr

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

    def _update_tree_state(self, opt_repr) -> None:
        """
        :param opt_repr: (batch, hid, seq)
        :param rhs_mask: (batch, seq)
        :return:
        """
        if self.tree_state_policy == "post_expansion":
            # tree_state: (batch, hid)
            tree_state = self._tree_state
            batch = opt_repr.size()[0]
            self._tree_state = self._tree_state_udpater(tree_state, opt_repr.unsqueeze(-1),
                                                        opt_repr.new_ones(batch, 1)).detach()

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
    def _retrieve_the_selected_option_repr(choice, opt_repr):
        """
        :param choice: (batch, )
        :param opt_repr: (batch, opt_num, hid)
        :return: (batch, hid)
        """
        batch_index = torch.arange(choice.size()[0], device=choice.device)
        return opt_repr[batch_index, choice]
