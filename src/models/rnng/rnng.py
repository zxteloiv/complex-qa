import logging
from typing import Union, Optional, Literal, List, Callable

import torch
from allennlp.training.metrics import Average
from torch import nn
from allennlp.modules import Embedding as AllenEmbedding
from trialbot.data import NSVocabulary

from models.interfaces.encoder import EncoderStack
from models.interfaces.unified_rnn import RNNStack, T_HIDDEN
from models.modules.batched_stack import TensorBatchStack
from models.modules.variational_dropout import VariationalDropout
from utils.nn import get_final_encoder_states, seq_cross_ent
from utils.seq_collector import SeqCollector


class RNNG(nn.Module):
    def __init__(self,
                 # modules
                 # an RNNStack requires an initial state, but an EncoderStack doesn't
                 action_encoder: RNNStack,      # encodes action history, from h0
                 buffer_encoder: RNNStack,      # encodes temporary tokens, from h0
                 stack_encoder: EncoderStack,   # encodes the stack
                 reducer: EncoderStack,         # compose the popped stack tokens
                 action_embedding: Union[nn.Embedding, AllenEmbedding],
                 action_projection: nn.Module,

                 # model configuration
                 nt_gen_id_boundary: int,
                 hidden_size: int,

                 start_id: int,
                 root_id: int,
                 padding_index: int = 0,
                 max_action_num: int = 200,

                 # training configuration
                 emb_drop: float = .1,
                 loss_reduction: Literal['token', 'batch'] = 'batch',
                 ):
        super(RNNG, self).__init__()
        self.embed = action_embedding
        self.reducer = reducer
        self.stack_encoder = stack_encoder
        self.buffer_encoder = buffer_encoder
        self.action_encoder = action_encoder
        self.projection = action_projection

        self.hidden_sz = hidden_size
        self.emb_drop = VariationalDropout(emb_drop, on_the_fly=False)
        self.emb_sz: int = action_embedding.weight.size()[-1]
        if self.hidden_sz != self.emb_sz:
            self.emb2hid = nn.Sequential(nn.Linear(self.emb_sz, self.hidden_sz), nn.Tanh())
        else:
            self.emb2hid = lambda x: x

        # 0    1    2       3  ...  boundary ...
        # PAD  OOV  REDUCE  NT ...  SHIFT    ...
        self.padding_id = padding_index
        self.start_id = start_id
        self.root_id = root_id
        self.boundary = nt_gen_id_boundary
        self.max_action_num = max_action_num
        self.loss_reduction = loss_reduction

        self._stack: Optional[TensorBatchStack] = None
        # because the stack could store a variety of data and we cannot tell types from merely embeddings,
        # we maintain another stack that simultaneously pushes or pops data with the embedding stack
        self._stack_tag: Optional[TensorBatchStack] = None
        self._tag_dim = 1   # a single indicator denoting openness is sufficient in RNNG
        self._buf: Optional[TensorBatchStack] = None
        self._attend_and_compose: Optional[Callable] = None
        self._initial_hx: Optional[T_HIDDEN] = None
        self._loss: Optional[torch.Tensor] = None

        self._action_len = Average()
        self._token_len = Average()
        self._err_rate = Average()

        self._check_dimensions()

    def _check_dimensions(self):
        # a reducer pushes a new item onto the stack
        assert self.reducer.get_output_dim() == self.hidden_sz
        # a reducer composes items from the stack
        assert self.reducer.get_input_dim() == self.hidden_sz
        # a stack encoder processes items from the stack
        assert self.stack_encoder.get_input_dim() == self.hidden_sz
        # the buffer and action encoder may be initialized by the same hidden states,
        # e.g. from the source token encoder
        assert self.buffer_encoder.get_output_dim() == self.action_encoder.get_output_dim()
        # action and buffer encoders deal with the projected embeddings (mapped to hidden sz)
        assert self.buffer_encoder.get_input_dim() == self.hidden_sz
        assert self.action_encoder.get_input_dim() == self.hidden_sz

    def set_conditions(self,
                       init_state: Optional[List[torch.Tensor]],
                       attend_and_compose_fn: Callable,
                       ) -> None:
        """
        For conditional generation (as in seq2rnng model), batch conditions are passed in
        1) to build init hx for action and buffer encoders;
        2) to build attentions for final word predictions;

        For the inputs to action, buffer, and stack encoders,
        we do not inject any attention mechanism although it is possible.

        :param init_state: [(batch, hidden)]
        :param attend_and_compose_fn: a function s.t. (h -> (h -> context) -> h')
        :return: None
        """
        self._initial_hx, _ = self.buffer_encoder.init_hidden_states(init_state)
        self._attend_and_compose = attend_and_compose_fn

    def forward(self,
                batch_sz: int,
                actions: Optional[torch.LongTensor],
                target_tokens: Optional[torch.LongTensor] = None,
                ):
        """
        Forward the RNNG generation.
        :param batch_sz: batch size of the given
        :param actions: (batch, max_action_num)
        :param target_tokens: (batch, max_seq_len)
        :return:
        """
        self.emb_drop.reset()
        batch_sz = self._check_args(batch_sz, actions, target_tokens)
        device = next(self.parameters()).device
        self._loss = None

        last_pred = self._init_iteration(batch_sz, device)
        action_hx = self._initial_hx
        acc_succ = torch.full((batch_sz,), 1, device=device)
        steps = range(self.get_step_num(actions))
        mem = SeqCollector()
        for step in steps:
            step_input = actions[:, step] if self.training else last_pred
            if step > 0:
                acc_succ *= (self._stack._top_cur > -1).long()  # avoid actions on empty stack
            action_hx, step_logit, acc_succ = self._forward_step(step_input, action_hx, acc_succ)
            last_pred = step_logit.argmax(dim=-1)
            mem(logit=step_logit, pred=last_pred)
            if acc_succ.sum() == 0 and not self.training:
                # no valid instances, stop the iterations immediately
                break

        logits = mem.get_stacked_tensor('logit')
        preds = mem.get_stacked_tensor('pred')
        if self.training:
            self.compute_loss(logits, actions)

        if target_tokens is not None:
            self.compute_metrics(actions, target_tokens, preds)
        return preds, logits

    def get_loss(self):
        return self._loss

    def _check_args(self, batch_sz, action_ids, oracle_tokens):
        assert batch_sz is not None or action_ids is not None
        batch_sz = batch_sz or action_ids.size()[0]
        if action_ids is not None and batch_sz != action_ids.size()[0]:
            raise ValueError("specified batch size does not math the action ids")

        if self.training:
            if action_ids is None:
                raise ValueError("Training requires the action sequence provided.")
        return batch_sz

    def _init_iteration(self, batch_sz: int, device, action_num: int = None):
        stack_sz = action_num or self.max_action_num
        self._buf = TensorBatchStack(batch_sz, stack_sz, self.hidden_sz, device=device)
        self._stack = TensorBatchStack(batch_sz, stack_sz, self.hidden_sz, device=device)
        self._stack_tag = TensorBatchStack(batch_sz, stack_sz, self._tag_dim,
                                           dtype=torch.long, device=device,)
        # start_emb: (batch, hidden_sz)
        start_emb = self.embed_actions(torch.full((batch_sz,), self.start_id, device=device).long())
        self._buf.push(start_emb, None)
        # root_ids
        root_ids = torch.full((batch_sz,), self.root_id, device=device).long()
        return root_ids

    def embed_actions(self, x: torch.LongTensor) -> torch.Tensor:
        return self.emb2hid(self.emb_drop(self.embed(x)))

    def get_step_num(self, action_ids: Optional):
        if action_ids is None:
            return self.max_action_num

        if self.training:
            return action_ids.size()[-1] - 1

        # To save computation time, instead of setting a max action number,
        # the action id length is also adopted for evaluation.
        return action_ids.size()[-1] - 1

    def _forward_step(self, last_action: torch.LongTensor, action_hx, acc_succ=None):
        # last_action, acc_suc: (batch,)
        acc_succ = acc_succ if acc_succ is not None else last_action.new_ones((last_action.size()[0],))
        # set action to 0 for the failed instances
        # because 0 is used for padding and will not interrupt the stacks,
        # and 0 is not an action so the stacks will be safe
        last_action = last_action * acc_succ

        # is_*: (batch,)
        is_reduce_action = (last_action == 2)
        is_nt_action = torch.logical_and(2 < last_action, last_action < self.boundary)
        is_gen_action = self.boundary <= last_action

        if is_reduce_action.any():
            acc_succ *= self._do_reduce_transactions(is_reduce_action.long())

        # action_emb: (batch, hidden)
        action_emb = self.embed_actions(last_action)
        if is_nt_action.any():
            acc_succ *= self._do_nt_transactions(action_emb, is_nt_action.long())

        if is_gen_action.any():
            acc_succ *= self._do_gen_transactions(action_emb, is_gen_action.long())

        buffer_embedding = self._encode_output_buffer(self._initial_hx)
        stack_embedding = self._encode_stack_buffer()
        action_hx, a_seq_embedding = self.action_encoder.forward(action_emb, action_hx)

        machine_state = torch.cat([stack_embedding, a_seq_embedding, buffer_embedding], dim=-1)
        attended_state = self._attend_and_compose(machine_state)
        step_logit = self.projection(attended_state)
        return action_hx, step_logit, acc_succ

    def _do_nt_transactions(self, action_emb, is_nt) -> torch.Tensor:
        # a transaction may be successful for some cases but not others
        succ = self._stack.push(action_emb, is_nt)
        batch_sz = action_emb.size()[0]
        succ = succ * self._stack_tag.push(is_nt.new_ones(batch_sz, 1).long(), is_nt * succ)
        return succ

    def _do_gen_transactions(self, action_emb, is_gen) -> torch.Tensor:
        succ = self._stack.push(action_emb, is_gen)
        batch_sz = action_emb.size()[0]
        succ = succ * self._stack_tag.push(is_gen.new_zeros(batch_sz, 1).long(), is_gen * succ)
        succ = succ * self._buf.push(action_emb, is_gen * succ)
        return succ

    def _do_reduce_transactions(self, is_reduce):
        mem = SeqCollector()
        initial_redicible = is_reduce.detach()
        acc_succ = self._pop_until_nonterminals(is_reduce, mem)

        emb = mem.get_stacked_tensor('emb')     # (batch, seq, hidden)
        mask = mem.get_stacked_tensor('mask')   # (batch, seq)
        # emb and mask will not be None because the transaction is only called when some is_reduce > 0,
        # but they may be still empty for some place.
        if (mask.sum() == 0).all():             # if the reduce transaction has failed for the entire batch,
            return torch.zeros_like(acc_succ)   # no need to check each flag anymore

        output = self.reducer(emb, mask)        # (batch, seq, hidden)

        # (batch, hidden)
        final_states = get_final_encoder_states(output, mask, self.reducer.is_bidirectional())
        acc_succ = acc_succ * self._stack.push(final_states, initial_redicible * acc_succ)
        return acc_succ

    def _pop_until_nonterminals(self, is_reduce, mem):
        acc_succ = is_reduce.new_ones(is_reduce.size()[0])
        while (is_reduce > 0).any():
            emb, succ = self._stack.pop(is_reduce * acc_succ)
            acc_succ = acc_succ * succ

            # unwanted locations have been already defaulted to be 0
            tags, succ = self._stack_tag.pop(is_reduce * acc_succ)
            acc_succ = acc_succ * succ

            # emb: (batch, hidden)
            # tags: (batch, 1)
            # acc_succ: (batch,)
            mem(emb=emb, tags=tags, mask=is_reduce * acc_succ)

            # pop should stopped when any opening tag (=1) is met
            is_reduce = is_reduce - tags[:, 0]
            # pop should stopped when any failure occurs
            is_reduce = is_reduce * acc_succ
        return acc_succ

    def _encode_output_buffer(self, hx):
        # token_emb: (batch, seq, hidden)
        # mask: (batch, seq)
        token_emb, mask = self._buf.dump()
        steps = range(mask.size()[-1])
        outputs = []
        for step in steps:
            hx, step_out = self.buffer_encoder.forward(token_emb[:, step], hx)
            outputs.append(step_out)
        # buffer_embedding: (batch, hidden)
        buffer_embedding = get_final_encoder_states(torch.stack(outputs, dim=1), mask)
        return buffer_embedding

    def _encode_stack_buffer(self):
        item_emb, mask = self._stack.dump()
        if mask.sum() > 0:
            output = self.stack_encoder.forward(item_emb, mask)
            stack_embedding = get_final_encoder_states(output, mask, self.stack_encoder.is_bidirectional())
        else:
            stack_embedding = item_emb.new_zeros(mask.size()[0], self.hidden_sz)
        return stack_embedding  # (batch, hidden)

    def compute_loss(self, logits, action_ids):
        gold_action = action_ids[:, 1:].contiguous()
        gold_mask = (action_ids != self.padding_id)[:, 1:].contiguous()
        self._loss = seq_cross_ent(logits, gold_action, gold_mask, self.loss_reduction)

    def compute_metrics(self, action_ids, oracle_tokens, preds):
        action_lengths = (action_ids != self.padding_id).sum(-1)
        for l in action_lengths:
            self._action_len(l)

        token_lengths = (oracle_tokens != self.padding_id).sum(-1)
        for l in token_lengths:
            self._token_len(l)

        for pred, oracle in zip(oracle_tokens, preds):
            pred_ids = list(filter(lambda x: x >= self.boundary, pred.tolist()))
            oracle_ids = list(filter(lambda x: x != self.padding_id, oracle.tolist()))
            if len(pred_ids) != len(oracle_ids):
                self._err_rate(1)
            elif any(x != y for x, y in zip(pred_ids, oracle_ids)):
                self._err_rate(1)
            else:
                self._err_rate(0)

    def get_metrics(self, reset=False):
        metrics = {
            "ERR": self._err_rate.get_metric(reset),
            "ALEN": self._action_len.get_metric(reset),
            "TLEN": self._token_len.get_metric(reset)
        }
        return metrics


