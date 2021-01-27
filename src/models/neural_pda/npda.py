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
from .partial_tree_encoder import TopDownTreeEncoder
from .tree import Tree
from .rule_scorer import RuleScorer
from utils.nn import init_state_for_stacked_rnn, get_final_encoder_states
from utils.nn import prepare_input_mask, expand_tensor_size_at_dim

from .tensor_typing_util import *

class NeuralPDA(nn.Module):
    def __init__(self,
                 # modules
                 symbol_embedding: nn.Embedding,
                 lhs_symbol_mapper,

                 grammar_tutor,
                 token_tutor,
                 rhs_expander: StackedRNNCell,

                 pre_tree_encoder: TopDownTreeEncoder,
                 pre_tree_self_attn: nn.Module,
                 residual_norm_after_self_attn: nn.Module,

                 rule_scorer: RuleScorer,

                 exact_token_predictor: nn.Module,

                 # configuration
                 grammar_entry: int,
                 max_derivation_step: int = 1000,
                 detach_tree_encoder: bool = False,
                 dropout: float = 0.2,
                 ):
        super().__init__()
        self._expander = rhs_expander
        self._gt = grammar_tutor
        self._tt = token_tutor
        self._embedder = symbol_embedding
        self._lhs_symbol_mapper = lhs_symbol_mapper
        self._exact_token_predictor = exact_token_predictor

        self._dropout = nn.Dropout(dropout)

        self._pre_tree_encoder = pre_tree_encoder
        self._pre_tree_self_attn = pre_tree_self_attn
        self._rule_scorer = rule_scorer
        self._residual_norm = residual_norm_after_self_attn

        # -------------------
        # configurations
        self.grammar_entry = grammar_entry
        self.max_derivation_step = max_derivation_step  # a very large upper limit for the runtime storage
        self.detach_tree_encoder = detach_tree_encoder

        # -------------------
        # the helpful storage for runtime forwarding
        self._query_attn_fn = None
        self._derivation_step = 0
        # the stack contains information about both the node position and the node token value
        self._stack: Optional[TensorBatchStack] = None
        self._partial_tree: Optional[Tree] = None
        self._tree_hx = None
        self.diagnosis = dict()

    def init_automata(self, batch_sz: int, device: torch.device, query_attn_fn: Callable,
                      max_derivation_step: int = 0):
        # init tree
        self._partial_tree = Tree(batch_sz, self.max_derivation_step * 5, 1, device=device)
        self._tree_hx = None

        root_id = torch.zeros((batch_sz, 1), device=device).long()
        root_val = torch.full((batch_sz, 1), fill_value=self.grammar_entry, device=device).long()
        self._partial_tree.init_root(root_val)

        root_item = torch.cat([root_id, root_val], dim=-1)

        # init stack
        self._stack = TensorBatchStack(batch_sz, self.max_derivation_step, 2, dtype=torch.long, device=device)
        self._stack.push(root_item, push_mask=None)

        # init derivation counter
        self._derivation_step = 0
        # init tree state
        # since the automata independently runs for every batch, the query attention is also initialized every batch.
        self._query_attn_fn = query_attn_fn

        if max_derivation_step > 0:
            self.max_derivation_step = max_derivation_step + 1

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
        self._query_attn_fn = None
        self._tree_hx = None
        self._partial_tree = None
        self._stack = None

    def forward(self, **kwargs):
        return self._forward_parallel_training(**kwargs)

    def _forward_parallel_training(self,
                                   tree_nodes,  # (batch, n_d), the tree nodes = #lhs = num_derivations
                                   node_parents,  # (batch, n_d),
                                   expansion_frontiers,  # (batch, n_d, max_runtime_stack_size),
                                   derivations,  # (batch, n_d, max_seq), the gold derivations for choice checking
                                   ):
        self.diagnosis = dict()
        _, tree_mask = prepare_input_mask(tree_nodes)

        # nodes_emb: (batch, n_d, emb)
        # tree_hid: (batch, n_d, hid)
        nodes_emb = self._embedder(tree_nodes)
        nodes_emb = self._lhs_symbol_mapper(nodes_emb)
        tree_hid, _ = self._pre_tree_encoder(nodes_emb, node_parents, tree_mask)
        if self.detach_tree_encoder:
            tree_hid = tree_hid.detach()

        batch, n_d = tree_nodes.size()
        attn_mask = tree_mask.new_zeros((batch, n_d, n_d))
        attn_mask[torch.arange(batch, device=tree_mask.device).reshape(-1, 1, 1),
                  torch.arange(n_d, device=tree_mask.device).reshape(1, -1, 1),
                  expansion_frontiers] = 1
        # the root has the id equal to the padding and must be excluded from attention targets except for itself.
        attn_mask[:, 1:, 0] = 0
        tree_hid_att = self._pre_tree_self_attn(tree_hid, tree_mask, attn_mask)
        if self._residual_norm is not None:
            tree_hid_att = tree_hid_att + tree_hid
            tree_hid_att = self._residual_norm(tree_hid_att)

        # tree_grammar: (batch, n_d, opt_num, 4, max_seq)
        tree_grammar = self._gt(tree_nodes)

        # attention computation and compose the context vector with the symbol hidden states
        # query_context: (batch, *, attention_out)
        query_context = self._query_attn_fn(tree_hid_att)

        # opt_prob: (batch, n_d, opt_num)
        # opt_repr: (batch, n_d, opt_num, hid)
        opt_prob, opt_repr = self._get_topological_choice_distribution(tree_nodes, tree_hid_att, query_context)

        exact_logit = self._predict_exact_token_after_derivation(derivations, tree_hid_att, query_context)

        self.diagnosis.update(tree_hid=tree_hid, tree_hid_att=tree_hid_att)
        return opt_prob, exact_logit, tree_grammar

    def _forward_derivation_step(self):
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
        # lhs_idx, lhs, lhs_mask: (batch,)
        # tree_state: (batch, hidden)
        tree_state = self._encode_partial_tree()    # tree encoding must precede the lhs popping
        lhs_idx, lhs, lhs_mask = self._select_node()

        # grammar_guide: (batch, opt_num, 4, max_seq)
        grammar_guide = self._gt(lhs)

        # attention computation and compose the context vector with the symbol hidden states
        # query_context: (batch, *, attention_out)
        query_context = self._query_attn_fn(tree_state)

        # opt_prob: (batch, opt_num)
        # opt_repr: (batch, opt_num, hid)
        opt_prob, opt_repr = self._get_topological_choice_distribution(lhs, tree_state, query_context)

        # best_choice: (batch,)
        best_choice = opt_prob.argmax(dim=-1)
        chosen_symbols, _, _ = self.retrieve_the_chosen_structure(best_choice, grammar_guide)
        exact_token_logit = self._predict_exact_token_after_derivation(chosen_symbols, tree_state, query_context)

        return lhs_idx, lhs_mask, grammar_guide, opt_prob, exact_token_logit

    def _encode_partial_tree(self) -> FT:
        # node_val: (batch, node_num, node_val_dim)
        # node_parent, node_mask: (batch, node_num)
        node_val, node_parent, node_mask = self._partial_tree.dump_partial_tree()
        node_val = node_val.squeeze(-1)

        # node_emb: (batch, node_num, emb_sz)
        node_emb = self._embedder(node_val)
        node_emb = self._lhs_symbol_mapper(node_emb)

        # tree_hidden: (batch, node_num, hidden_sz)
        tree_hidden, tree_hx = self._pre_tree_encoder(node_emb, node_parent, node_mask, self._tree_hx)
        self._tree_hx = tree_hx

        batch_index = torch.arange(self._stack.max_batch_size, device=node_val.device).long()
        # leaf_mask should be consistent with node_mask
        # top_item: (batch, 2)
        # top_pos: (batch,)
        top_item, top_mask = self._stack.top()
        top_pos = top_item[:, 0] * top_mask

        # encode all leaves, all the current node except for those who had been others' parent nodes.
        # attn_mask: (batch, node_num)
        attn_mask = node_mask
        # if node_mask.size()[-1] > 1:
        # # stack nodes & mask: (batch, max_stack_num, 2) & (batch, max_stack_num)
        # stack_nodes, stack_mask = self._stack.dump()
        # stack_pos = stack_nodes[:, :, 0] * stack_mask
        # attn_mask[batch_index.unsqueeze(-1), stack_pos] = 0
        attn_mask[batch_index.unsqueeze(-1), node_parent] = 0
        attn_mask[batch_index, top_pos] = 1 # the node itself must be involved into self-attention
        tree_hid_att = self._pre_tree_self_attn(tree_hidden, attn_mask)
        if self._residual_norm is not None:
            tree_hid_att = tree_hid_att + tree_hidden
            tree_hid_att = self._residual_norm(tree_hid_att)

        # tree_state: (batch, hid)
        tree_state = tree_hid_att[batch_index, top_pos]
        return tree_state

    def _select_node(self) -> Tuple[LT, LT, LT]:
        node, success = self._stack.pop()
        node_idx = node[:, 0]
        node_val = node[:, 1]
        return node_idx, node_val, success

    def _get_topological_choice_distribution(self, lhs, tree_state, query_context):
        """
        get the topological choice distribution for the current derivation, conditioning on the given LHS and Tree States
        :param lhs: (batch, *)
        :param tree_state: (batch, *, hid_dim)
        :param query_context: (batch, *, attention_dim)
        :return: tuple of 2:
                (batch, *, opt_num) choice probabilities,
                (batch, *, opt_num, hid) compositional representation of each rule option
        """

        # full_grammar_repr: (V, opt_num, hid)
        # full_grammar_mask: (V, opt_num)
        full_grammar_repr, full_grammar_mask = self._get_full_grammar_repr()

        # opt_repr: (batch, *, opt_num, hid)
        # opt_mask: (batch, *, opt_num)
        opt_repr = full_grammar_repr[lhs]
        opt_mask = full_grammar_mask[lhs]

        # opt_logit, opt_prob: (batch, *, opt_num)
        opt_num = opt_mask.size()[-1]
        exp_qc = expand_tensor_size_at_dim(query_context, opt_num, dim=-2)
        exp_ts = expand_tensor_size_at_dim(tree_state, opt_num, dim=-2)
        opt_logit = self._rule_scorer(opt_repr, exp_qc, exp_ts)
        opt_prob = masked_softmax(opt_logit, opt_mask.bool())

        return opt_prob, opt_repr

    def _get_full_grammar_repr(self):
        (
            valid_symbols,      # valid_symbols: (V, opt_num, max_seq)
            parental_growth,    # p_growth: (V, opt_num, max_seq)
            fraternal_growth,   # f_growth: (V, opt_num, max_seq)
            symbol_mask,        # symbol_mask: (V, opt_num, max_seq)
        ) = self.decouple_grammar_guide(self._gt._g)

        num_symbols, opt_num, max_symbol_len = valid_symbols.size()

        # lhs symbol is all the symbol vocabulary
        # (V,)
        lhs_symbol = torch.arange(num_symbols, device=valid_symbols.device)

        # init_state: (V, hid)
        init_state = self._lhs_symbol_mapper(self._dropout(self._embedder(lhs_symbol)))
        hx, _ = self._expander.init_hidden_states(
            init_state_for_stacked_rnn([init_state], self._expander.get_layer_num(), "all")
        )

        mem = SeqCollector()
        for step in range(0, max_symbol_len):
            # prepare ranking features
            step_index = torch.tensor([step], device=valid_symbols.device)
            step_symbol = valid_symbols.index_select(dim=-1, index=step_index).squeeze(-1)
            step_p_growth = parental_growth.index_select(dim=-1, index=step_index)
            step_f_growth = fraternal_growth.index_select(dim=-1, index=step_index)
            same_as_lhs = (step_symbol == lhs_symbol.unsqueeze(-1)).unsqueeze(-1)

            emb = self._dropout(self._embedder(step_symbol))  # (V, opt_num, emb_sz)
            # step_output: (V, opt_num, hid)
            hx, step_out = self._expander(emb, hx, input_aux=[step_p_growth, step_f_growth, same_as_lhs])
            mem(step_output=step_out)

        # rhs_output: (V, opt_num, max_seq, hid)
        rhs_output = mem.get_stacked_tensor('step_output', dim=-2)
        # opt_repr: (V, opt_num, hid)
        opt_repr = get_final_encoder_states(
            rhs_output.reshape(num_symbols * opt_num, max_symbol_len, -1),
            symbol_mask.reshape(num_symbols * opt_num, max_symbol_len)
        ).reshape(num_symbols, opt_num, -1)

        # (V, opt_num)
        opt_mask = symbol_mask.sum(-1) > 0
        return opt_repr, opt_mask

    def _predict_exact_token_after_derivation(self, symbols: LT, tree_state: FT, query_context: FT) -> FT:
        """
        :param symbols: (batch, *, max_seq)
        :param tree_state: (batch, *, hid)
        :param query_context: (batch, *, attention_dim)
        :return:
        """
        # compliant_weights: (batch, *, max_seq, V)
        compliant_weights = self._tt(symbols).bool()

        # symbol_emb: (batch, *, max_seq, emb)
        symbol_emb = self._dropout(self._embedder(symbols))

        seq_size = symbol_emb.size()[-2]
        exp_ts = expand_tensor_size_at_dim(tree_state, seq_size, dim=-2)
        exp_qc = expand_tensor_size_at_dim(query_context, seq_size, dim=-2)
        proj_inp = torch.cat([symbol_emb, exp_ts, exp_qc], dim=-1)
        # exact_logit: (batch, *, V)
        exact_logit = self._exact_token_predictor(proj_inp)
        # exact_logit = exact_logit + (compliant_weights + tiny_value_of_dtype(exact_logit.dtype)).log()
        exact_logit = exact_logit.masked_fill(~compliant_weights, min_value_of_dtype(exact_logit.dtype))
        return exact_logit

    def update_pda_with_predictions(self, parent_idx: LT, symbol: LT, p_growth: LT, symbol_mask: LT, lhs_mask: NullOrLT):
        """
        Iterate the symbols backwards and push them onto the stack based on topological information.

        :param parent_idx: (batch,)
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
            step_symbol = symbol[:, step].unsqueeze(-1)
            push_mask = push_mask.long()

            # node_idx, succ: (batch,)
            node_idx, succ = self._partial_tree.add_new_node_edge(step_symbol, parent_idx, push_mask)
            stack_item = torch.cat([node_idx.unsqueeze(-1), step_symbol], dim=-1)
            self._stack.push(stack_item, (push_mask * succ).long())

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
    def decouple_grammar_guide(grammar: LT):
        ndim = grammar.ndim
        index_pref = [slice(None, None)] * (ndim - 2)
        index_suffix = [slice(None, None)] * 1
        splitted = []
        for i in range(4):
            index = index_pref + [i] + index_suffix
            splitted.append(grammar[index])

        (
            valid_symbols,      # valid_symbols: (batch, *, opt_num, max_seq)
            parental_growth,    # p_growth: (batch, *, opt_num, max_seq)
            fraternal_growth,   # f_growth: (batch, *, opt_num, max_seq)
            symbol_mask,        # symbol_mask: (batch, *, opt_num, max_seq)
        ) = splitted
        return valid_symbols, parental_growth, fraternal_growth, symbol_mask

    @staticmethod
    def find_choice_by_symbols(gold_symbols: LT, valid_symbols: LT, symbol_mask: LT) -> LT:
        """
        :param gold_symbols: (batch, *, max_derivation_len)
        :param valid_symbols: (batch, *, opt_num, max_possible_len)
        :param symbol_mask: (batch, *, opt_num, max_possible_len)
        :return:
        """
        size_bound = max(gold_symbols.size()[-1], valid_symbols.size()[-1])
        def _pad(t):
            if t.size()[-1] == size_bound:
                return t
            return F.pad(t, [0, size_bound - t.size()[-1]], value=0)

        gold_mask = (gold_symbols != 0).long()
        gold_symbols, gold_mask, valid_symbols, symbol_mask = [
            _pad(t) for t in (gold_symbols, gold_mask, valid_symbols, symbol_mask)
        ]

        # seq_comp: (batch, *, opt_num, max_len)
        # the comparison actually requires symbol_a * mask_a == symbol_b * mask_b,
        # but the symbols are all properly masked here now.
        seq_comp = (valid_symbols * symbol_mask == (gold_symbols * gold_mask).unsqueeze(-2))
        # only the non-empty sequences require comparison, other choices default to 0
        # _choice: (batch, *, opt_num)
        _choice = seq_comp.all(dim=-1) * (symbol_mask.sum(-1) > 0)
        # there must be one and only one choice found for all batch, otherwise the data is inconsistent
        assert (_choice.sum(dim=-1) == (gold_mask.sum(-1) > 0)).all()

        # argmax is forbidden on bool storage, but max is ok
        return _choice.max(dim=-1)[1]

