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

        # the helpful storage for runtime forwarding
        self._query_attn_fn = None
        self._tree_state = None
        self._derivation_step = 0

    def forward(self, token_based=False, **kwargs):
        if not token_based:
            return self._forward_derivation_step(**kwargs)

        else:
            raise NotImplementedError

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

    def _forward_derivation_step(self, step_symbols=None):
        """
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

        opts_logp = self._expand_rhs(lhs, tree_state, grammar_guide, step_symbols=step_symbols)
        topo_preds = self.inference_tree_topology(opts_logp, grammar_guide)
        if step_symbols is None:
            inferred_symbols = topo_preds[0]
        else:
            inferred_symbols = step_symbols[:, :grammar_guide.size()[-1]]

        exact_token_logit = self._predict_exact_token_after_derivation(inferred_symbols)
        return lhs, lhs_mask, grammar_guide, opts_logp, topo_preds, exact_token_logit

    def continue_derivation(self):
        if self._derivation_step >= self.max_derivation_step:
            return False
        self._derivation_step += 1

        _, success = self._stack.top()
        batch_stack_is_not_empty = success.sum() > 0
        return batch_stack_is_not_empty

    def _select_node(self) -> Tuple[LT, LT]:
        node, success = self._stack.pop()
        return node.squeeze(-1).long(), success

    def _encode_partial_tree(self):
        return self._tree_state.detach() if self._tree_state is not None else None

    def _expand_rhs(self, lhs, tree_state, grammar_guide, step_symbols: NullOrLT) -> FT:

        mem = SeqCollector()
        step_inp = lhs
        for step in range(grammar_guide.size()[-1]):
            if step > 0 and step_symbols is not None:
                step_inp = step_symbols[:, step - 1]
            # Produce a locally-compliant step output and symbol distribution.
            # step_output: (batch, hid)
            # s_logits: (batch, V)
            step_output, symbol_logits = self._predict_tree_symbol(step_inp, tree_state)

            self._update_partial_tree(step_output)

            # bias the logits with the step grammar
            # symbol_opts, opt_mask: (batch, opts)
            # compliant_logits: (batch, V)
            symbol_opts, opt_mask = grammar_guide[:, :, 0, step], grammar_guide[:, :, -1, step]
            compliant_logits = self._learn_from_step_grammar_tutor(symbol_logits, symbol_opts, opt_mask)
            if step_symbols is None:
                step_inp = compliant_logits.argmax(dim=-1)

            mem(symbol_logits=compliant_logits)

        # seq_logits, seq_logp: (batch, V, seq)
        seq_logits = mem.get_stacked_tensor('symbol_logits', dim=-1)
        seq_logp = (F.softmax(seq_logits, dim=1) + 1e-13).log()

        # seq_opts, opt_mask: (batch, opt_num, seq_len)
        # seq_opts_logp: (batch, opt_num, seq_len)
        seq_opts, opt_mask = grammar_guide[:, :, 0, :], grammar_guide[:, :, -1, :]
        seq_opts_logp = torch.gather(seq_logp, index=seq_opts, dim=1)

        # opts_logp: (batch, opt_num)
        opts_logp = (seq_opts_logp * opt_mask).sum(dim=-1)
        return opts_logp

    def _predict_tree_symbol(self, symbol, tree_state) -> Tuple[FT, FT]:
        # symbol: (batch,)
        # emb: (batch, emb)
        emb = self._dropout(self._embedder(symbol))

        # tree_state: (batch, hidden)
        # the hidden state is discarded each time, hidden inputs of the RNN is constructed from the tree
        hx = None
        if tree_state is not None:
            hx, _ = self._expander.init_hidden_states([tree_state for _ in range(self._expander.get_layer_num())])
        _, step_out = self._expander(emb, hx)

        # attention computation and compose the context vector with the symbol hidden states
        query_context = self._query_attn_fn(step_out)
        step_out_att = self._query_attn_comp(query_context, step_out)

        # s_logits: (batch, V)
        symbol_logits = self._symbol_predictor(step_out_att)
        return step_out_att, symbol_logits

    def _update_partial_tree(self, incremental) -> None:
        self._tree_state = self._tree_state_udpater(self._tree_state, incremental)

    @staticmethod
    def _learn_from_step_grammar_tutor(logit, compliance, compliance_mask) -> FT:
        """
        Adjust the symbol logits at current step.
        Grammar Guide Mask is not required yet, but it will be used in the entire sequence.

        :param logit: (batch, V), the symbol logits at the current step
        :param compliance: (batch, max_opts), the marginalized compliance at the current step
        :return:
        """
        # the default weight is 0; set all weights for the positions contained in opts to 1;
        # reset the weights of 0 index with valid 0s
        compliance = compliance * compliance_mask
        weights = torch.zeros_like(logit).bool()
        weights[torch.arange(logit.size()[0], device=logit.device).unsqueeze(-1), compliance] = 1
        weights[:, 0] = (((compliance == 0) * compliance_mask).sum(-1) > 0) # the mask is ignored at the current step
        masked_logit = logit + (weights + tiny_value_of_dtype(logit.dtype)).log()
        return masked_logit

    def _predict_exact_token_after_derivation(self, symbols: LT) -> FT:
        # symbol: (batch, seqlen)
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

    def push_predictions_onto_stack(self, topology_prediction: T3L, lhs_mask: LT):
        symbol, p_growth, f_growth = topology_prediction

        # step 0 is never added to the stack
        for step in range(symbol.size()[-1] - 1, 0, -1):
            push_mask = lhs_mask * p_growth[:, step] * f_growth[:, step - 1]
            assert push_mask.ndim == 1
            self._stack.push(symbol[:, step].unsqueeze(-1), push_mask.long())

    @staticmethod
    def inference_tree_topology(opts_logp, grammar_guide) -> T4L:
        """
        Inference the tree topological structure of a derivation.
        Exact token inference is not considered here.
        :param opts_logp: (batch, opt_num)
        :param grammar_guide: (batch, opt_num, 4, max_seq)
        :param step_symbols: (batch, max_derivation_len), Nullable
        :return: Tuple[(batch, seq) * 4], symbol sequence, parental growth, fraternal growth, and mask respectively.
        """
        # best_opt: (batch,)
        best_opt = opts_logp.argmax(dim=-1)

        # topo_pred: (batch, 4, max_seq)
        topology_predictions = grammar_guide[torch.arange(best_opt.size()[0], device=best_opt.device), best_opt]

        symbol = topology_predictions[:, 0, :]
        p_growth = topology_predictions[:, 1, :]
        f_growth = topology_predictions[:, 2, :]
        mask = topology_predictions[:, 3, :]

        return symbol, p_growth, f_growth, mask
