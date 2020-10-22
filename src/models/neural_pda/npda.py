from typing import Optional, Callable, Tuple, Generator, Mapping
import torch
from torch import nn
import torch.nn.functional as F
from .npda_cell import PDACellBase, NPDAHidden
from .batched_stack import BatchStack
from .nt_decoder import NTDecoder
import logging, datetime
from trialbot.data.ns_vocabulary import NSVocabulary
from utils.seq_collector import SeqCollector

class NeuralPDA(torch.nn.Module):
    def __init__(self,
                 pda_decoder: PDACellBase,
                 nt_decoder: NTDecoder,
                 batch_stack: BatchStack,
                 num_nonterminals: int,
                 nonterminal_dim: int,
                 token_embedding: nn.Embedding,
                 token_predictor: nn.Module,
                 start_token_id: int,
                 padding_token_id: Optional[int] = None,
                 validator: Optional[nn.Module] = None,
                 init_codebook_confidence: int = 1,  # the number of iterations from which the codebook is learnt
                 codebook_training_decay: float = 0.99,
                 ntdec_init_policy: str = "next_token",
                 fn_decode_token: Optional[Callable[[int], str]] = None,
                 ):
        super().__init__()
        self.pda_decoder = pda_decoder
        self.nt_decoder = nt_decoder

        self.num_nt = num_nonterminals
        self.nonterminal_dim = nonterminal_dim
        codebook = F.normalize(torch.randn(num_nonterminals, nonterminal_dim)).detach()
        self.codebook = nn.Parameter(codebook)
        self._code_m_acc = nn.Parameter(codebook.clone())
        self._code_n_acc = nn.Parameter(torch.full((num_nonterminals,), init_codebook_confidence))
        self._code_decay = codebook_training_decay

        self.token_embedding = token_embedding
        self.token_predictor = token_predictor

        self.ntdec_init_policy = ntdec_init_policy

        self.validator = validator
        self.stack = batch_stack

        self.pad_id = padding_token_id
        self.start_id = start_token_id
        self.logger = logging.getLogger(__name__)
        self.fn_decode_token = fn_decode_token

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
                    raw_codes: (batch, step, 2, hidden_dim)
                    valid_logits: (batch, step, 3), or None if validator is not required
        """
        batch_size = batch_size or x.size()[0]
        decoding_length = x.size()[1] if x is not None else max_generation_len
        device = default_device or (x.device if x is not None else None)
        x_mask = x_mask or (x != self.pad_id).long()
        self.logger.debug(f"forward parameters: batch={batch_size}, "
                          f"length={decoding_length}, device={device}")

        # initialize the NPDA model.
        self._init_stack(batch_size, default_device=device)
        mem_bread = SeqCollector()

        last_preds = torch.full((batch_size,), self.start_id, device=device)
        acc_m, acc_n = None, None

        self.logger.debug(f'start npda decoding: {datetime.datetime.now().strftime("%M:%S.%f")[:-3]}')
        for step in range(decoding_length):
            self.logger.debug(f'--> get input {datetime.datetime.now().strftime("%M:%S.%f")[:-3]}')
            # step_input: (batch, hidden_dim)
            step_tok = last_preds if x is None else x[:, step]
            # top: (batch, hidden_dim)
            # top_mask: (batch,)
            top, top_mask = self.stack.pop()
            # step_mask indicating whether the current iteration is held accountable within the batch,
            # either empty stack or padding input of x will lead to a violation of the update.
            step_mask = x_mask[:, step] * top_mask

            self.logger.debug(f'--> forward step {datetime.datetime.now().strftime("%M:%S.%f")[:-3]}')
            # logits: (batch, vocab)
            # codes: (batch, 2, hidden_dim)
            # valid_logits: (batch, 3) or None
            h, logits, codes, valid_logits = self._forward_step(step_tok, top, h, attn_fn=attn_fn)
            last_preds = torch.argmax(logits, dim=-1)
            mem_bread(logits=logits, codes=codes, inp_token=step_tok, inp_stack=top, inp_mask=step_mask)

            self.logger.debug(f'--> quantize {datetime.datetime.now().strftime("%M:%S.%f")[:-3]}')
            # As long as the stack is empty, no more item could be pushed onto it.
            # quantized_code: (batch_size, 2, hidden_dim)
            # quantized_idx, push_mask: (batch_size, 2)
            quantized_code, quantized_idx = self._quantize_code(codes)
            self.logger.debug(f'--> stack push {datetime.datetime.now().strftime("%M:%S.%f")[:-3]}')
            push_mask = quantized_idx * step_mask.unsqueeze(-1)         # code indices are happened to be push masks
            self.stack.push(quantized_code[:, 0, :], push_mask[:, 0])   # mask>0 is equivalent to mask=1 by convention
            self.stack.push(quantized_code[:, 1, :], push_mask[:, 1])
            mem_bread(push=push_mask, valid=valid_logits, qnt_idx=quantized_idx)

            if self.training:
                self.logger.debug(f'--> moving avg {datetime.datetime.now().strftime("%M:%S.%f")[:-3]}')
                m_t, n_t = self._get_step_moving_averages(codes, quantized_idx, step_mask)
                acc_m = m_t if acc_m is None else acc_m + m_t
                acc_n = n_t if acc_n is None else acc_n + n_t

        self.logger.debug(f'end decoding: {datetime.datetime.now().strftime("%M:%S.%f")[:-3]}')
        # update the moving average counter, but do not update the codebook which
        if self.training:
            r = self._code_decay
            self._code_m_acc = nn.Parameter(r * self._code_m_acc + (1 - r) * acc_m)
            self._code_n_acc = nn.Parameter(r * self._code_n_acc + (1 - r) * acc_n)

        # logits: (batch, step, vocab)
        # pushes: (batch, step, 2)
        # raw_codes: (batch, step, 2, hidden_dim)
        # valid_logits: (batch, step, 3)
        tlogits = mem_bread.get_stacked_tensor('logits')
        pushes = mem_bread.get_stacked_tensor('push').long()
        raw_codes = mem_bread.get_stacked_tensor('codes')
        vlogits = mem_bread.get_stacked_tensor('valid') if self.validator is not None else None
        return tlogits, pushes, raw_codes, vlogits, mem_bread

    def update_codebook(self):
        """Called when the codebook parameters need update, after the optimizer update step perhaps"""
        if self.training:
            self.codebook = nn.Parameter(self._code_m_acc / (self._code_n_acc.unsqueeze(-1) + 1e-15))

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

        last_token = None
        if self.ntdec_init_policy == "current_token":
            last_token = step_input
        elif self.ntdec_init_policy == "last_state_default_none":
            last_token = None if h is None else h.state
        elif self.ntdec_init_policy == "last_state_default_current":
            last_token = step_input if h is None else h.state

        h = self.pda_decoder(step_input, stack_top, h)

        # any thing in h_all: (batch, hidden_dim)
        h_all = h.token, h.stack, h.state  # terminal, non-terminal, and state of automata
        if attn_fn is not None:
            h_all = (h + attn_fn(h) for h in h_all)
            if last_token is not None:
                last_token = last_token + attn_fn(last_token)
                last_token = last_token.tanh()
        h_t, h_nt, h_s = [h.tanh() for h in h_all]

        if self.ntdec_init_policy == "next_state":
            last_token = h_t

        # logits: (batch, vocab)
        # codes: (batch, 2, hidden_dim)
        # valid_logits: (batch, 3) or None
        logits = self.token_predictor(h_t)
        codes = self.nt_decoder(h_nt, last_token)
        valid_logits = self.validator(h_s) if self.validator is not None else None

        return h, logits, codes, valid_logits

    def _quantize_code(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param codes: (batch_size, -1, hidden_dim)
        :return: a tuple of
                (batch_size, n, hidden_dim), the quantized code vector, selected directly from the codebook
                (batch_size, n), the id of the code in the codebook
        """
        # codes: (B, n, hidden_dim)
        batch_sz, _, hidden_dim = codes.size()

        # code_rs: (nB, 1, hidden_dim)
        # self.codebook: (#NT, hidden_dim) -> (1, #NT, hidden_dim)
        # diff_vec: (nB, #NT, hidden_dim)
        code_rs = codes.reshape(-1, 1, hidden_dim)
        diff_vec = code_rs - torch.unsqueeze(self.codebook, dim=0)

        # diff_norm: (nB, #NT)
        # quantized_idx: (nB,)
        diff_norm = torch.norm(diff_vec, dim=2)
        quantized_idx = torch.argmin(diff_norm, dim=1)

        quantized_codes = self.codebook[quantized_idx]
        return quantized_codes.reshape(batch_sz, -1, hidden_dim), quantized_idx.reshape(batch_sz, -1)

    def _init_stack(self, batch_size: int, default_device: Optional[torch.device] = None):
        stack_bottom = torch.zeros((batch_size, self.nonterminal_dim), device=default_device)
        push_mask = torch.ones((batch_size,), device=default_device)
        self.stack.reset(batch_size, default_device=default_device)
        self.stack.push(stack_bottom, push_mask)

    def tracing_generation(self, collector: SeqCollector, gold: Optional[torch.Tensor] = None):
        """
        collector: { tags: [ seq-tag-val ] }

        ExecTrace -> { BatchReplays GlobalStat }
        BatchReplays -> [ ReplayPerItem* ]
        ReplayPerItem -> { [ Step* ] StackLoad }
        Step -> { Prod PredNext GoldNext Errno }
        Prod -> S w A B  # implies a single rule S -> w A B
        Errno -> TokenMask PDAMask

        :return:
        """
        resource_tags = ('inp_token', 'inp_stack', 'inp_mask', 'logits', 'qnt_idx')
        resources = list(map(collector.get_stacked_tensor, resource_tags))
        # inp_token, inp_mask: (batch, steps)
        # inp_stack: (batch, steps, hidden_dim)
        # tlogits: (batch, steps, #tokens)
        # qnt_idx: (batch, steps, 2)
        inp_token, inp_stack, inp_mask, tlogits, qnt_idx = resources

        # stack_load: [batch]
        stack_stat = self.stack.describe() # { str: [batch] }
        stack_load = stack_stat['stack_load']

        # inp_qnt_idx: (batch, steps)
        # preds: (batch, steps)
        inp_qnt_idx = self._quantize_code(inp_stack)[1]
        preds = tlogits.argmax(dim=-1)

        batch_size, step_num = inp_token.size()
        batch_replays = [
            {
                "final_load": stack_load[batch_id],
                "steps": [
                    {
                        "prod": [inp_qnt_idx[batch_id][step_id].item(),
                                 inp_token[batch_id][step_id].item()
                                 ] + qnt_idx[batch_id][step_id].tolist(),
                        "pred": preds[batch_id][step_id].item(),
                        "gold": gold[batch_id][step_id].item() if gold is not None else 0,
                        "mask": inp_mask[batch_id][step_id].item()
                    }
                    for step_id in range(step_num)
                ],
            }
            for batch_id in range(batch_size)
        ]

        decode = self.fn_decode_token
        if decode is not None:
            for replay in batch_replays:
                for step in replay['steps']:
                    ntid, tid, nt2, nt1 = step['prod']  # swap order due to the FILO stack
                    next_tid, next_gold = step['pred'], step['gold']
                    nt0, nt1, nt2 = [f"<{nt}>" for nt in (ntid, nt1, nt2)]
                    token, next_token, gold_token = [decode(t) for t in [tid, next_tid, next_gold]]
                    step['decode_prod'] = [nt0, token, nt1, nt2]
                    step['decode_pred'] = next_token
                    step['decode_gold'] = gold_token

        return batch_replays

