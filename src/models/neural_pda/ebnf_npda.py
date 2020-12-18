from typing import Optional, Callable, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from .batched_stack import BatchStack
import logging
from utils.seq_collector import SeqCollector
from utils.nn import get_final_encoder_states, get_decoder_initial_states
from models.modules.stacked_rnn_cell import StackedRNNCell
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.nn.util import masked_softmax

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
                 dropout: float = 0.,
                 default_max_derivation: int = 100,
                 default_max_expansion: int = 10,
                 topology_predictor: Optional[nn.Module] = None,
                 derivation_hist_similairty: Optional[MatrixAttention] = None,
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

        self.logger = logging.getLogger(__name__)
        self.dropout = nn.Dropout(dropout)

        self.default_max_derivation: int = default_max_derivation
        self.default_max_expansion: int = default_max_expansion

        self.topology_pred = topology_predictor
        self.derivation_hist_similairty = derivation_hist_similairty
        self._derivation_mapping = None
        if derivation_hist_similairty is not None:
            self._derivation_mapping = nn.Linear(ebnf_expander.hidden_dim, ebnf_expander.hidden_dim)

    def forward(self,
                derivation_tree: Optional[torch.LongTensor] = None,
                is_nt: Optional[torch.LongTensor] = None,
                inp_mask: Optional[torch.LongTensor] = None,
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
        :param inp_mask: (batch, derivation, rhs_seq), useful when init the hidden states.
        :param max_expansion_len: int,
        :param max_derivation_num: int,
        :param batch_size: overwritten batch size, must be specified if npda_start is absent.
        :param device: overwritten device of npda, must be specified if npda_start is absent.
        :param hx:
        :param attn_fn:
        :param parallel_mode:
        :return: is_nt_prob: (batch, derivation, rhs_seq)
                nt_logits: (batch, derivation, rhs_seq, #NT)
                t_logits: (batch, derivation, rhs_seq, #T)
        """
        batch_size = batch_size or derivation_tree.size()[0]
        device = device or derivation_tree.device

        self._init_stack(derivation_tree, batch_size, device)
        if parallel_mode and is_nt is not None:
            self.logger.debug("Run with parallel mode to independently predict each derivation of every instance")
            return self._forward_parallel(derivation_tree, is_nt, attn_fn, mask=inp_mask)

        elif parallel_mode:
            # only the LHS is given (derivation length == 1), the RHS is unknown (is_nt is None)
            # to some extent is equivalent to generate given oracle tree structure
            return self._forward_parallel(derivation_tree, None, attn_fn, max_expansion_len)

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

    def _forward_serial(self, derivation_count, expansion_len):
        for derive_step in range(derivation_count):
            pass
        raise NotImplementedError

    def _forward_parallel(self, derivation_tree,
                          is_nt: Optional[torch.LongTensor] = None,
                          attn_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                          expansion_len: Optional[int] = None,
                          mask: Optional[torch.LongTensor] = None,
                          ):
        # Forward pass when the oracle derivation_tree or at least the oracle LHS is given,
        # and in the meantime all the derivations are independent with each other,
        # therefore hx is prohibited.
        # By "Parallel" it means to expand all the grammar LHS simultaneously.
        # So the is_nt flag could be absent, and if so, the expansion is greedily generated.
        # Typical scenarios are either (tree-)language model or grammar warm-up.
        # Note is_nt doesn't contain LHS info.
        tree_shape = derivation_tree.size()  # (batch, derivation, seq)
        flat_derivations = self._flatten_derivation(derivation_tree)

        flat_lhs = flat_derivations[:, 0]
        flat_rhs_seq = flat_is_nt = None
        if is_nt is not None:
            flat_rhs_seq = flat_derivations[:, 1:]
            flat_is_nt = self._flatten_derivation(is_nt)

        attn_fn = self._build_attn_for_flat_out(tree_shape[0], attn_fn)
        # the hx is meaningful and useful for training, only encoder attention is used.
        # although it will work for inference,
        # it had better use serial inference to propagate the complete tree state
        hx = self._init_parallel_hx(flat_lhs, flat_is_nt, flat_rhs_seq, attn_fn, mask)

        # out: (batch*derivation, hidden_dim)
        flat_out = self.expander.get_output_state(hx)
        out = self._revert_flat_derivation(flat_out, tree_shape[0])
        dec_attn_fn = self._build_runtime_dec_hist_attn_fn(tree_shape[0], out)

        def combined_attn(out):
            context = 0
            if attn_fn is not None:
                context = context + attn_fn(out)

            if dec_attn_fn is not None:
                context = context + dec_attn_fn(out)
            return context

        flat_predictions = self._expand_rhs(flat_lhs, flat_is_nt, flat_rhs_seq,
                                            hx=hx, attn_fn=combined_attn, max_expansion_len=expansion_len)

        # flat_is_nt: (batch*derivation, rhs_seq)
        # flat_nt_logits: (batch*derivation, rhs_seq, #NT)
        # flat_t_logits: (batch*derivation, rhs_seq, #T)
        flat_is_nt, flat_nt_logits, flat_t_logits, _ = flat_predictions
        is_nt_prob = self._revert_flat_derivation(flat_is_nt, tree_shape[0])
        nt_logits = self._revert_flat_derivation(flat_nt_logits, tree_shape[0])
        t_logits = self._revert_flat_derivation(flat_t_logits, tree_shape[0])
        return is_nt_prob, nt_logits, t_logits

    @staticmethod
    def _flatten_derivation(t):
        return t.reshape(-1, *t.size()[2:])

    @staticmethod
    def _revert_flat_derivation(t, batch_sz):
        return t.reshape(batch_sz, -1, *t.size()[1:])

    def _build_attn_for_flat_out(self, batch_sz, attn):
        # due to the parallel setting, any EBNF embedding output is flattened,
        # which will make the attention fail because it's unable to recognize the batch size
        if attn is None:
            return None

        def attn_for_flat_out(flat_out):
            context = attn(self._revert_flat_derivation(flat_out, batch_sz))
            return self._flatten_derivation(context)
        return attn_for_flat_out

    def _build_runtime_dec_hist_attn_fn(self, batch_sz: int, history):
        """
        :param batch_sz:
        :param history: (batch, hist_derivation, hidden_dim)
        :return:
        """
        if self.derivation_hist_similairty is None:
            return None

        def attn_for_flat_out(flat_out):
            # for the serial inference the derivation is 1, so everything would be just fine
            # flat_out: (batch*derivation, hidden)
            # out: (batch, derivation, hidden)
            out = flat_out.reshape(batch_sz, -1, flat_out.size()[-1])
            # similarity: (batch, derivation, hist_derivation)
            similarity = self.derivation_hist_similairty(out, history)

            # in parallel mode, derivation == hist_derivation
            derivation, hist_derivation = similarity.size()[1:]
            mask = torch.ones_like(similarity[0])
            if derivation == hist_derivation:   # mask will be conducted
                mask = mask.tril_(-1)

            # mask: (1, derivation, hist_derivation)
            mask = mask.unsqueeze(0)
            # attn_weights: (batch, derivation, hist_derivation)
            attn_weights = masked_softmax(similarity, mask, dim=-1)

            # context: (batch, derivation, hidden_dim)
            context = torch.matmul(attn_weights, history)
            return self._flatten_derivation(self._derivation_mapping(context))

        return attn_for_flat_out

    def _init_parallel_hx(self, lhs, is_nt: Optional, gold_rhs: Optional,
                          attn_fn: Optional,
                          mask: Optional = None,
                          batch_sz: Optional[int] = None):
        batch_sz = batch_sz or lhs.size()[0]
        with torch.no_grad():
            # out_emb: (batch*derivation, rhs_seq, hidden_dim)
            _, _, _, out_emb = self._expand_rhs(lhs, is_nt, gold_rhs, attn_fn=attn_fn)
            # mask: (-1, rhs_seq)
            if mask is not None:
                if mask.ndim >= out_emb.ndim:
                    mask = mask.reshape(-1, mask.size()[-1])
                crude_out = get_decoder_initial_states([out_emb], mask, num_decoder_layers=self.expander.get_layer_num())
                del mask
            else:
                crude_out = [out_emb[:, -1, :] for _ in range(self.expander.get_layer_num())]

            one_step_context = []
            for layer_out in crude_out:
                layer_out = torch.roll(layer_out, 1, dims=0)
                layer_out = layer_out.reshape(batch_sz, -1, *layer_out.size()[1:])
                layer_out[:, 0, :] = 0
                layer_out = layer_out.reshape(-1, layer_out.size()[-1])
                one_step_context.append(layer_out)

            hidden, _ = self.expander.init_hidden_states(one_step_context)
            del _, crude_out, one_step_context
            return hidden

    def _expand_rhs(self,
                    lhs: Optional[torch.LongTensor],
                    nt_mask: Optional[torch.LongTensor] = None,
                    target_symbol: Optional[torch.LongTensor] = None,
                    max_expansion_len: Optional[int] = None,
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
        decoding_len = nt_mask.size()[1] if nt_mask is not None else (max_expansion_len or self.default_max_expansion)
        mem = SeqCollector()

        # Prepare the first input token of the RHS, some terminal like <GO>,
        # which is required when the target symbol sequence is not given.
        if target_symbol is None or nt_mask is None:
            last_preds = torch.full((batch_size,), fill_value=self.start_id, dtype=torch.int, device=device)
            last_is_nt = torch.full_like(last_preds, fill_value=1)

        # lhs_emb: (batch, emb_sz)
        lhs_emb = self.dropout(self.emb_nonterminals(lhs))

        for step in range(decoding_len):
            # step_tok: (batch,)
            # tok_is_nt: (batch,)
            # step_emb: (batch, emb_sz)
            if target_symbol is None or nt_mask is None:
                step_tok, tok_is_nt = last_preds, last_is_nt
            else:
                step_tok, tok_is_nt = target_symbol[:, step], nt_mask[:, step]

            step_emb = self._get_batch_symbol_embedding(step_tok, tok_is_nt)

            hx, out_emb, out_is_nt, nt_logits, t_logits = self._pred_step(lhs_emb, tok_is_nt, step_emb, hx, attn_fn)
            mem(is_nt_prob=out_is_nt, nt_logits=nt_logits, t_logits=t_logits, step_out=out_emb)

            # inference
            if target_symbol is None or nt_mask is None:
                last_is_nt = out_is_nt > 0.5
                last_is_t = last_is_nt.logical_not()
                last_preds = (nt_logits.argmax(dim=-1).mul_(last_is_nt).add_(
                    t_logits.argmax(dim=-1).mul_(last_is_t))
                ).long()

        t_logits = mem.get_stacked_tensor('t_logits')
        nt_logits = mem.get_stacked_tensor('nt_logits')
        is_nt_prob = mem.get_stacked_tensor('is_nt_prob')
        step_out = mem.get_stacked_tensor('step_out')
        return is_nt_prob, nt_logits, t_logits, step_out

    def _pred_step(self, lhs_emb, tok_is_nt, step_emb, hx=None, attn_fn=None):
        # LHS is concatenated to every step input
        # step_inp: (batch, emb_sz + 1 + emb_sz)
        step_inp = torch.cat([lhs_emb, tok_is_nt.unsqueeze(-1), step_emb], dim=-1)

        # step_out: (batch, decoder_hidden)
        hx, step_out = self.expander(step_inp, hx)
        if attn_fn is not None:
            context = attn_fn(step_out)
            step_out = (step_out + self.dropout(context)).tanh()

        if self.topology_pred is None:
            # step_out_is_nt: (batch,)
            step_out_is_nt = torch.sigmoid(step_out[:, 0])

            # *_logits: (batch, #count)
            step_out_emb = self.dropout(step_out[:, 1:])
            nt_logits = self.nt_predictor(step_out_emb)
            t_logits = self.t_predictor(step_out_emb)
        else:
            step_out_emb = self.dropout(step_out)
            step_out_is_nt = self.topology_pred(step_out_emb).squeeze(-1)
            nt_logits = self.nt_predictor(step_out_emb)
            t_logits = self.t_predictor(step_out_emb)

        return hx, step_out_emb, step_out_is_nt, nt_logits, t_logits

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
        step_emb_nt = self.emb_nonterminals(nt_tok_safe.long())
        step_emb_t = self.emb_terminals(t_tok_safe.long())
        step_emb = step_emb_nt * tok_is_nt + step_emb_t * tok_is_nt.logical_not()
        step_emb = self.dropout(step_emb)
        return step_emb

