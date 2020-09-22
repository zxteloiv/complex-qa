from typing import Optional, Callable, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from .npda_cell import RNNPDACell, NPDAHidden
from .batched_stack import BatchedStack
from .nt_decoder import NTDecoder

class NeuralPDA(torch.nn.Module):
    def __init__(self,
                 pda_decoder: RNNPDACell,
                 nt_decoder: NTDecoder,
                 batch_stack: BatchedStack,
                 num_nonterminals: int,
                 num_terminals: int,
                 nonterminal_dim: int,
                 token_embedding: nn.Embedding,
                 token_predictor: nn.Module,
                 start_token_id: int,
                 padding_token_id: Optional[int] = None,
                 validator: Optional[nn.Module] = None,
                 init_codebook_confidence: int = 1,     # the number of iterations from which the codebook is learnt
                 ):
        super().__init__()
        self.pda_decoder = pda_decoder
        self.nt_decoder = nt_decoder

        self.num_terminal = num_terminals
        self.num_nt = num_nonterminals
        codebook = F.normalize(torch.randn(num_nonterminals, nonterminal_dim)).detach()
        self.codebook = nn.Parameter(codebook)
        self._code_m_acc = nn.Parameter(codebook.clone())
        self._code_n_acc = nn.Parameter(torch.full((num_nonterminals,), init_codebook_confidence))
        self._code_decay = 0.99

        self.token_embedding = token_embedding
        self.token_predictor = token_predictor

        self.validator = validator
        self.stack = batch_stack

        self._pad_id = padding_token_id
        self._start_id = start_token_id

    def forward(self,
                x: Optional[torch.LongTensor] = None,
                x_mask: Optional[torch.LongTensor] = None,
                h: Optional[NPDAHidden] = None,
                attn_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                batch_size: Optional[int] = None,
                max_generation_len: int = 100,
                default_device: Optional[torch.device] = None,
                ):
        """
        Run the model.
        If x is given, the model will run for the likelihood via teacher forcing.
        Otherwise, the model will generate the token sequence and its likelihood.
        The batch size is either manually set or depending on the x, but they must be identical if both are available.
        The max_generation_len is only useful when x is not given.

        :param x: (batch, length), the input states
        :param x_mask: (batch, length), the pre-assigned token masks indicating the sequence in a batch
        :param h: Initial State for the Current State.
        :param attn_fn: A closure calling dynamically to attend over other data source.
                        nothing would happen if None is specified.
        :param batch_size: an integer indicating the batch size, guiding the generation when `x` is absent.
        :param max_generation_len: an integer indicating the number of generation steps when `x` is absent.
        :param default_device: device to initialize tensors which overrides the device of `x`.
        :return: a tuple of token logits, push records(non-differentiable), raw_nt_codes, validation logits if any
                    logits: (batch, step, vocab)
                    pushes: (batch, step, 2)
                    valid_logits: (batch, step, 3), or None if validator is not required
        """
        batch_size = batch_size or x.size()[0]
        decoding_length = x.size()[1] if x is not None else max_generation_len
        device = default_device or (x.device if x is not None else None)
        x_mask = x_mask or (x != self._pad_id).long()

        # initialize the NPDA model.
        self._init_stack(batch_size, default_device=device)
        logits_by_step, push_by_step, code_by_step, valid_by_step = [], [], [], []
        last_preds = torch.full((batch_size,), self._start_id, device=device)
        acc_m, acc_n = None, None

        for step in range(decoding_length):
            # step_input: (batch, hidden_dim)
            step_tok = last_preds if x is None else x[:, step]
            # top: (batch, hidden_dim)
            # top_mask: (batch,)
            top, top_mask = self.stack.pop()
            # step_mask indicating whether the current iteration is held accountable within the batch,
            # either empty stack or padding input of x will lead to a violation of the update.
            step_mask = x_mask[:, step] * top_mask

            # logits: (batch, vocab)
            # codes: (batch, 2, hidden_dim)
            # valid_logits: (batch, 3) or None
            h, logits, codes, valid_logits = self._forward_step(step_tok, top, h, attn_fn=attn_fn)
            logits_by_step.append(logits)
            last_preds = torch.argmax(logits, dim=-1)
            code_by_step.append(codes)

            # As long as the stack is empty, no more item could be pushed onto it.
            # quantized_code: (batch_size, 2, hidden_dim)
            # quantized_idx, push_mask: (batch_size, 2)
            quantized_code, quantized_idx = self._quantize_code(codes)
            push_mask = quantized_idx * step_mask.unsqueeze(-1)         # code indices are happened to be push masks
            self.stack.push(quantized_code[:, 0, :], push_mask[:, 0])   # mask>0 is equivalent to mask=1 by convention
            self.stack.push(quantized_code[:, 1, :], push_mask[:, 1])
            push_by_step.append(push_mask)
            valid_by_step.append(valid_logits)

            if self.training:
                m_t, n_t = self._get_step_moving_averages(codes, quantized_idx, step_mask)
                acc_m = m_t if acc_m is None else acc_m + m_t
                acc_n = n_t if acc_n is None else acc_n + n_t

        # update the moving average counter, but do not update the codebook which
        if self.training:
            r = self._code_decay
            self._code_m_acc = r * self._code_m_acc + (1 - r) * acc_m
            self._code_n_acc = r * self._code_n_acc + (1 - r) * acc_n

        # logits: (batch, step, vocab)
        # pushes: (batch, step, 2)
        # raw_codes: (batch, step, 2, hidden_dim)
        # valid_logits: (batch, step, 3)
        logits = torch.stack(logits_by_step, dim=1)
        pushes = torch.stack(push_by_step, dim=1)
        raw_codes = torch.stack(code_by_step, dim=1)
        valid_logits = torch.stack(valid_by_step, dim=1) if self.validator is not None else None
        return logits, pushes, raw_codes, valid_logits

    def update_codebook(self):
        """Called when the codebook parameters need update, after the optimizer update step perhaps"""
        if self.training:
            self.codebook.copy_(self._code_m_acc / (self._code_n_acc + 1e-15))

    def _get_step_moving_averages(self,
                                  codes: torch.Tensor,
                                  quantized_idx: torch.Tensor,
                                  updating_mask: torch.Tensor):
        """
        Compute and return the moving average accumulators (m and n).
        :param codes: (batch, 2, hidden_dim)
        :param quantized_idx: (batch, 2)
        :param updating_mask: (batch,)
        :return: (#NT, d), (#NT) the accumulator of each timestep for the nominator and denominators.
        """
        # qidx_onehot, filtered_qidx: (batch, 2, #NT)
        qidx_onehot = (quantized_idx.unsqueeze(-1) == torch.arange(self.num_nt, device=codes.device).reshape(1, 1, -1))
        filtered_qidx = updating_mask.reshape(-1, 1, 1) * qidx_onehot

        # n_t: (#NT,) the number of current codes quantized for each NT
        # m_t: (#NT, d)
        n_t = filtered_qidx.sum(0).sum(0)
        m_t = (filtered_qidx.unsqueeze(-1) * codes.unsqueeze(-2)).sum(0).sum(0)

        return m_t, n_t

    def _forward_step(self, step_tok, stack_top, h, attn_fn=None):
        step_input = self.token_embedding(step_tok)
        h = self.pda_decoder(step_input, stack_top, h)

        # any thing in h_all: (batch, hidden_dim)
        h_all = h.token, h.stack, h.state  # terminal, non-terminal, and state of automata
        if attn_fn is not None:
            h_all = (h + attn_fn(h) for h in h_all)
        h_t, h_nt, h_s = [h.tanh() for h in h_all]

        # logits: (batch, vocab)
        # codes: (batch, 2, hidden_dim)
        # valid_logits: (batch, 3) or None
        logits = self.token_predictor(h_t)
        codes = self.nt_decoder(h_nt, h_t)
        valid_logits = self.validator(h_s) if self.validator is not None else None

        return h, logits, codes, valid_logits

    def _quantize_code(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param codes: (batch_size, 2, hidden_dim)
        :return: a tuple of
                (batch_size, 2, hidden_dim), the quantized code vector, selected directly from the codebook
                (batch_size, 2), the id of the code in the codebook
        """
        # codes: (B, 2, hidden_dim)
        batch_sz, _, hidden_dim = codes.size()

        # code_rs: (2B, 1, hidden_dim)
        # self.codebook: (#NT, hidden_dim) -> (1, #NT, hidden_dim)
        # diff_vec: (2B, #NT, hidden_dim)
        code_rs = codes.reshape(batch_sz * 2, 1, hidden_dim)
        diff_vec = code_rs - torch.unsqueeze(self.codebook, dim=0)

        # diff_norm: (2B, #NT)
        # quantized_idx: (2B, 1)
        diff_norm = torch.norm(diff_vec, dim=2)
        quantized_idx = torch.argmin(diff_norm, dim=1, keepdim=True)

        quantized_codes = torch.gather(self.codebook, 1, quantized_idx)
        return quantized_codes.reshape(batch_sz, 2, hidden_dim), quantized_idx.reshape(batch_sz, 2)

    def _init_stack(self, batch_size: int, default_device: Optional[torch.device] = None):
        stack_bottom = torch.zeros((batch_size, self.nonterminal_dim), device=default_device)
        push_mask = torch.ones((batch_size,), device=default_device)
        self.stack.reset(batch_size)
        self.stack.push(stack_bottom, push_mask)
