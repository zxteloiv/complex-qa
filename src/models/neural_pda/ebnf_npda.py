from typing import Optional, Callable, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from .batched_stack import BatchStack
import logging
from utils.seq_collector import SeqCollector
from models.modules.stacked_rnn_cell import StackedRNNCell

class NeuralEBNF(nn.Module):
    def __init__(self,
                 emb_nonterminals,
                 emb_terminals,
                 num_nonterminals: int,
                 ebnf_expander: StackedRNNCell,
                 state_transition,
                 batch_stack: BatchStack,
                 predictor_nonterminals,
                 predictor_terminals,
                 start_token_id: int,
                 ebnf_entrypoint: int,
                 padding_token_id: int = 0,
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.emb_nonterminals = emb_nonterminals
        self.num_nt = num_nonterminals
        self.emb_terminals = emb_terminals
        self.expander = ebnf_expander
        self.stack = batch_stack
        self.transition_maker = state_transition
        self.nt_predictor = predictor_nonterminals
        self.t_predictor = predictor_terminals
        self.start_id = start_token_id
        self.grammar_entry = ebnf_entrypoint
        self.padding_id = padding_token_id

        self.logger = logging.getLogger(__name__)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                derivation_tree: Optional[torch.LongTensor] = None,
                is_nt: Optional[torch.LongTensor] = None,
                max_expansion_len: int = 10,
                max_derivation_num: int = 20,
                batch_size: Optional[int] = None,
                device: Optional[torch.device] = None,
                hx=None,
                attn_fn=None,
                parallel_mode: bool = True,
                ):
        """

        :param derivation_tree: (batch, derivation, seq), each seq starts with an LHS symbol.
        :param is_nt: (batch, derivation, rhs_seq), rhs_seq = seq - 1, excluding the LHS symbol.
        :param max_expansion_len: int,
        :param max_derivation_num: int,
        :param batch_size: overwritten batch size, must be specified if npda_start is absent.
        :param device: overwritten device of npda, must be specified if npda_start is absent.
        :param hx:
        :param attn_fn:
        :param parallel_mode:
        :return:
        is_nt_prob: (batch*derivation, rhs_seq)
        nt_logits: (batch*derivation, rhs_seq, #NT)
        t_logits: (batch*derivation, rhs_seq, #T)
        """
        batch_size = batch_size or derivation_tree.size()[0]
        device = device or derivation_tree.device

        self._init_stack(derivation_tree, batch_size, device)
        if parallel_mode and derivation_tree is not None:
            self.logger.debug("Run with parallel mode to independently predict each derivation of every instance")
            return self._forward_parallel(derivation_tree, is_nt)

        raise NotImplementedError


    def _init_stack(self, derivation_tree: Optional[torch.LongTensor], batch_size: int, device):
        # Init the stack with the given start non-terminals, which is the LHS of the first derivation.
        # If derivation tree is absent, the default entry of the grammar will be used.
        if derivation_tree is not None:
            start = derivation_tree[:, 0, 0]
        else:
            start = torch.full((batch_size,), fill_value=self.grammar_entry, device=device)

        symbol_is_nt = torch.ones((batch_size,), device=device)
        self.stack.reset(batch_size, device)
        stack_bottom = torch.stack([symbol_is_nt, start], dim=-1)
        push_mask = torch.ones((batch_size,), device=device)
        self.stack.push(stack_bottom, push_mask)

    def _forward_parallel(self, derivation_tree, is_nt):
        """
        Forward pass when derivation_tree is definitely given,
        and in the meantime all the derivations are independent with each other,
        therefore hx is prohibited.
        Typical scenarios are either (tree-)language model or grammar warm-up.
        Note is_nt doesn't contain LHS info.
        """
        tree_shape = derivation_tree.size() # (batch, derivation, seq)
        flat_derivations = derivation_tree.reshape(-1, tree_shape[-1])

        flat_lhs = flat_derivations[:, 0]
        flat_rhs_seq = flat_derivations[:, 1:]
        flat_is_nt = is_nt.reshape(-1, tree_shape[-1] - 1)
        flat_predictions = self._expand_rhs(flat_lhs, flat_is_nt, flat_rhs_seq)

        # flat_is_nt: (batch*derivation, rhs_seq)
        # flat_nt_logits: (batch*derivation, rhs_seq, #NT)
        # flat_t_logits: (batch*derivation, rhs_seq, #T)
        flat_is_nt, flat_nt_logits, flat_t_logits = flat_predictions

        is_nt_prob = flat_is_nt.reshape(*tree_shape[:-1], -1)
        nt_logits = flat_nt_logits.reshape(*is_nt.size(), -1)
        t_logits = flat_t_logits.reshape(*is_nt.size(), -1)

        return is_nt_prob, nt_logits, t_logits

    def _forward_serial(self):
        raise NotImplementedError

    def _expand_rhs(self,
                    lhs: Optional[torch.LongTensor],
                    nt_mask: Optional[torch.LongTensor] = None,
                    target_symbol: Optional[torch.LongTensor] = None,
                    max_expansion_len: int = 10,
                    hx = None,
                    attn_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                    ):
        """
        Expand the RHS with given LHS in the following agenda.
        1. Start with a start symbol for the LHS of an EBNF rule.
        2. Run the ebnf_expander up to some length for the RHS symbols (a for-loop)
        3. Decode the output of each RHS symbol, yielding predictions over both NT and T symbol tables.
        4. return the likelihoods.

        :param lhs: (batch,), the nonterminal id at the LHS
        :param nt_mask: (batch, seq), 0 or 1, nonterminals are denoted as 1
        :param target_symbol: (batch, seq), the symbol ids of the RHS seq (LHS is not included),
                some are non-terminals and others are terminals, based on the nt_mask
        :param max_expansion_len:
        :return: Tuple of the three: is_nt(batch, seq), nt_logits, t_logits (batch, seq, #NT/#T)
        """
        batch_size = lhs.size()[0]
        device = lhs.device
        decoding_len = nt_mask.size()[1] if nt_mask is not None else max_expansion_len
        mem = SeqCollector()

        # Prepare the first input token of the RHS, some terminal like <GO>,
        # which is required when the target symbol sequence is not given.
        last_preds = torch.full((batch_size,), fill_value=self.start_id, dtype=torch.int, device=device)
        last_is_nt = torch.full_like(last_preds, fill_value=0)

        # lhs_emb: (batch, emb_sz)
        lhs_emb = self.dropout(self.emb_nonterminals(lhs))

        for step in range(decoding_len):
            # step_tok: (batch,)
            # tok_is_nt: (batch,)
            # step_emb: (batch, emb_sz)
            step_tok = last_preds if target_symbol is None else target_symbol[:, step]
            tok_is_nt = last_is_nt if nt_mask is None else nt_mask[:, step]
            step_emb = self._get_batch_symbol_embedding(step_tok, tok_is_nt)

            # LHS is concatenated to every step input
            # step_inp: (batch, emb_sz + 1 + emb_sz)
            step_inp = torch.cat([lhs_emb, tok_is_nt.unsqueeze(-1), step_emb], dim=-1)

            # step_out: (batch, decoder_hidden)
            hx, step_out = self.expander(step_inp, hx)
            if attn_fn is not None:
                context = attn_fn(step_out)
                step_out = (step_out + context).tanh()

            # step_out_is_nt: (batch,)
            step_out_is_nt = torch.sigmoid(step_out[:, 0])

            # step_*_logits: (batch, #count)
            step_out_emb = self.dropout(step_out[:, 1:])
            nt_logits = self.nt_predictor(step_out_emb)
            t_logits = self.t_predictor(step_out_emb)

            mem(is_nt_prob=step_out_is_nt, nt_logits=nt_logits, t_logits=t_logits)

            # inference
            last_is_nt = step_out_is_nt > 0.5
            last_preds = nt_logits.argmax(dim=-1) * last_is_nt + \
                         t_logits.argmax(dim=-1) * last_is_nt.logical_not()

        t_logits = mem.get_stacked_tensor('t_logits')
        nt_logits = mem.get_stacked_tensor('nt_logits')
        is_nt_prob = mem.get_stacked_tensor('is_nt_prob')
        return is_nt_prob, nt_logits, t_logits

    def _get_batch_symbol_embedding(self, tok: torch.LongTensor, is_nt: torch.LongTensor):
        """
        A batch of tokens consists of both terminals and non-terminals,
        so their embeddings must be retrieved carefully.
        Embedding size of terminals and non-terminals are the same.
        :param tok: (batch,)
        :param is_nt: (batch,)
        :return: (batch, emb_sz)
        """
        nt_tok_safe = tok * is_nt
        t_tok_safe = tok * is_nt.logical_not()

        # tok_is_nt: (batch, 1) <- (batch,)
        tok_is_nt = is_nt.unsqueeze(-1)
        # step_emb*: (batch, emb_sz)
        step_emb_nt = self.emb_nonterminals(nt_tok_safe)
        step_emb_t = self.emb_terminals(t_tok_safe)
        step_emb = step_emb_nt * tok_is_nt + step_emb_t * tok_is_nt.logical_not()
        step_emb = self.dropout(step_emb)
        return step_emb

