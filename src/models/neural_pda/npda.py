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

from .tensor_typing_util import *

class NeuralPDA(nn.Module):
    def __init__(self,
                 # modules
                 symbol_embedding: nn.Embedding,

                 grammar_tutor,
                 # partial_tree_encoder,
                 # node_selector: nn.Module,
                 rhs_expander: StackedRNNCell,
                 stack: BatchStack,
                 parental_predictor: nn.Module,
                 fraternal_predictor: nn.Module,
                 symbol_predictor: nn.Module,
                 exact_form_predictor: nn.Module,

                 query_attention_composer: VectorContextComposer,

                 # configuration
                 grammar_entry: int,
                 max_derivation_step: int = 1000,
                 ):
        super().__init__()
        self._expander = rhs_expander
        self._stack = stack
        self._gt = grammar_tutor
        self._embedder = symbol_embedding
        self._symbol_predictor = symbol_predictor
        self._parental_predictor = parental_predictor
        self._fraternal_predictor = fraternal_predictor
        self._exact_form_predictor = exact_form_predictor

        self._query_attn_comp = query_attention_composer

        # configurations
        self.grammar_entry = grammar_entry
        self.max_derivation_step = max_derivation_step  # a very large upper limit for the runtime storage

        # the helpful storage for runtime forwarding
        self._query_attn_fn = None
        self._tree_state = None
        self._derivation_step = 0

        self.mem_prob_keys = ("symbol_p", "parental_p", "fraternal_p", "exact_symbol_p")

    def forward(self,
                token_based=False,
                ):
        if self.training:
            return self._forward_derivation_step()

        else:
            return self._forward_derivation_inference()

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

    def reset_automata(self):
        # init derivation counter
        self._derivation_step = 0
        # init tree state
        self._tree_state = None
        self._query_attn_fn = None

    def _forward_derivation_inference(self):
        mem = SeqCollector()
        while self.continue_derivation():
            rhs_p, lhs_mask = self._forward_derivation_step()
            predictions = self.inference(rhs_p)
            self.push_predictions_onto_stack(predictions, lhs_mask)
            mem(probability=rhs_p, prediction=predictions, lhs_mask=lhs_mask)

        return mem['probability'], mem['prediction'], mem['lhs_mask']

    def _forward_derivation_step(self):
        # lhs, lhs_mask: (batch,)
        # tree_state: (batch, hidden)
        lhs, lhs_mask = self._select_node()
        tree_state = self._encode_partial_tree()

        rhs_p = self._expand_rhs(lhs, tree_state)
        return rhs_p, lhs_mask

    def continue_derivation(self):
        if self._derivation_step >= self.max_derivation_step:
            return False
        self._derivation_step += 1

        _, success = self._stack.top()
        batch_stack_is_not_empty = success.sum() > 0
        return batch_stack_is_not_empty

    def _select_node(self) -> Tuple[Tensor, Tensor]:
        node, success = self._stack.pop()
        return node.squeeze(-1).long(), success

    def _encode_partial_tree(self):
        return self._tree_state.detach() if self._tree_state is not None else None

    def _expand_rhs(self, lhs, tree_state) -> T4F:
        # grammar_guide: (batch, max_num, 4, max_seq)
        grammar_guide = self._gt(lhs)

        mem = SeqCollector()
        step_inp = lhs
        for step in range(grammar_guide.size()[-1]):
            # step_output: (batch, hid)
            # s_logits: (batch, V)
            # p/f_logits: (batch,)
            step_output, step_logits = self._predict_tree_symbol(step_inp, tree_state)
            # step_grammar: List[(batch, max_num)]
            step_grammar = map(lambda x: x.squeeze(), grammar_guide[:, :, :, step].split(1, dim=-1))
            # topo_probs: Tuple[(batch, V), (batch,), (batch,)]
            topology_probs = self._learn_from_tutor(step_logits, step_grammar)
            exact_symbol_p = self._predict_exact_form(topology_probs[0].argmax(dim=-1))
            step_inp = topology_probs[0].argmax(dim=-1)
            mem(**dict(zip(self.mem_prob_keys, topology_probs + (exact_symbol_p,))))

        return tuple(map(mem.get_stacked_tensor, self.mem_prob_keys))

    def _predict_tree_symbol(self, last_symbol, tree_state) -> Tuple[FT, T3F]:
        # last_symbol: (batch,)
        # last_emb: (batch, emb)
        last_emb = self._embedder(last_symbol)

        # tree_state: (batch, hidden)
        # the hidden state is discarded each time, hidden inputs of the RNN is constructed from the tree
        hx = None
        if tree_state is not None:
            hx, _ = self._expander.init_hidden_states([tree_state for _ in range(self._expander.get_layer_num())])
        _, step_out = self._expander(last_emb, hx)

        # attention computation and compose the context vector with the symbol hidden states
        query_context = self._query_attn_fn(step_out)
        step_out_att = self._query_attn_comp(query_context, step_out)

        # s_logits: (batch, V)
        # p_logits: (batch,)
        # f_logits: (batch,)
        symbol_logits = self._symbol_predictor(step_out_att)
        parental_logits = self._parental_predictor(step_out_att).squeeze(-1)
        fraternal_logits = self._fraternal_predictor(step_out_att).squeeze(-1)
        self._update_partial_tree(step_out_att)

        return step_out_att, (symbol_logits, parental_logits, fraternal_logits)

    def _update_partial_tree(self, rule) -> None:
        if self._tree_state is None:
            self._tree_state = rule
        else:
            self._tree_state = self._tree_state + rule

    def _learn_from_tutor(self, step_logits, step_grammar) -> T3F:
        # step_logits: Tuple[(batch, V), (batch,), (batch,)]
        symbol_logits, parental_logits, fraternal_logits = step_logits
        # step_grammar: List[(batch, max_num)]
        symbol_opts, parental_growth, fraternal_mask = step_grammar

        symbol_prob = self._index_masked_prob(symbol_logits, symbol_opts, fraternal_mask)
        parental_prob = self._index_masked_prob(parental_logits, parental_growth, fraternal_mask)
        fraternal_prob = self._index_masked_prob(fraternal_logits, fraternal_mask, fraternal_mask)
        return symbol_prob, parental_prob, fraternal_prob

    @staticmethod
    def _index_masked_prob(logit, index, index_mask) -> FT:
        if logit.ndim > 1:
            # the default weight is 0; set all weights for the positions contained in opts to 1;
            # reset the weights of 0 index with valid 0s
            weights = torch.zeros_like(logit)
            weights[torch.arange(logit.size()[0], device=weights.device).unsqueeze(-1), index] = 1
            weights[:, 0] = (((index == 0) * index_mask).sum(-1) > 0)
            prob = masked_softmax(logit, weights.bool(), memory_efficient=True)

        else:
            logit += 10 * (((index == 1) * index_mask).sum(-1) > 0)
            logit -= 10 * (((index == 0) * index_mask).sum(-1) > 0)
            prob = logit.sigmoid()

        return prob

    def _predict_exact_form(self, symbol) -> FT:
        # symbol_emb: (batch, emb)
        symbol_emb = self._embedder(symbol)
        tree_state = self._encode_partial_tree()
        proj_inp = torch.cat([symbol_emb, tree_state], dim=-1)
        exact_p = F.softmax(self._exact_form_predictor(proj_inp), dim=-1)
        return exact_p

    def push_predictions_onto_stack(self, predictions: T4L, lhs_mask: NullOrT):
        symbol, p_growth, f_mask, exact_symbol = predictions
        mask = f_mask if lhs_mask is None else f_mask * lhs_mask.unsqueeze(-1)

        for step in range(symbol.size()[-1] - 1, -1, -1):
            push_mask = mask[:, step] * p_growth[:, step]
            self._stack.push(symbol[:, step].unsqueeze(-1), push_mask.long())

    @staticmethod
    def inference(rhs_p) -> T4L:
        # *symbol_p: (batch, seq, V)
        # p/f*_p: (batch, seq)
        symbol_p, parental_p, fraternal_p, exact_symbol_p = rhs_p

        # all predictions: (batch, seq)
        symbol = symbol_p.argmax(dim=-1)
        p_growth = (parental_p > 0.5).long()
        f_mask = (fraternal_p > 0.5).long().cumprod(dim=-1)
        exact_symbol = exact_symbol_p.argmax(dim=-1)

        return symbol, p_growth, f_mask, exact_symbol
