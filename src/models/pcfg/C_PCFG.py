from typing import Optional, Tuple

import torch
import torch.nn as nn
from allennlp.nn.util import masked_max

from .base_pcfg import PCFGMixin, PCFGModule
from .res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from ..modules.batched_stack import TensorBatchStack

from ..interfaces.encoder import EmbedAndEncode


def unit_norm(x, p=2, eps=1e-12):
    return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


class CompoundPCFG(PCFGModule):
    def __init__(self,
                 num_nonterminal: int,
                 num_preterminal: int,
                 num_vocab_token: int,
                 hidden_sz: int,
                 emb_enc: EmbedAndEncode = None,
                 z_dim: Optional[int] = None,
                 encoding_dim: int = None,
                 padding_id: int = 0,
                 ):
        super().__init__()

        self.NT = num_nonterminal
        self.T = num_preterminal
        self.V = num_vocab_token
        self.padding = padding_id

        self.hidden_sz = hidden_sz
        self.z_dim = z_dim or hidden_sz
        self.encoding_dim = encoding_dim or hidden_sz

        self.term_emb = nn.Parameter(torch.randn(self.T, self.hidden_sz))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.hidden_sz))
        self.root_emb = nn.Parameter(torch.randn(1, self.hidden_sz))

        compound_sz = self.hidden_sz + self.z_dim

        self.term_mlp = nn.Sequential(nn.Linear(compound_sz, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      nn.Linear(self.hidden_sz, self.V))

        self.root_mlp = nn.Sequential(nn.Linear(compound_sz, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      nn.Linear(self.hidden_sz, self.NT))

        self.NT_T = self.NT + self.T
        self.rule_mlp = nn.Linear(compound_sz, self.NT_T ** 2)

        # re-use the RNN and embeddings of the inference net, to build a new
        self.emb_enc = emb_enc
        if emb_enc is not None:
            self.infer_net = nn.Linear(emb_enc.get_output_dim(), self.z_dim * 2)

            # compose embeddings according to unary and binary rules
            self.unary_composer_word = nn.Sequential(nn.Linear(emb_enc.get_output_dim(), encoding_dim),
                                                     ResLayer(encoding_dim, encoding_dim))
            self.unary_composer_term = ResLayer(encoding_dim, encoding_dim)
            self.binary_composer_left = ResLayer(encoding_dim, encoding_dim)
            self.binary_composer_right = ResLayer(encoding_dim, encoding_dim)
            self.binary_composer_head = ResLayer(encoding_dim, encoding_dim)

        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def encode(self, x: torch.Tensor):
        assert x is not None
        # q_*: (batch, z_dim)
        x_hid, q_mean, q_logvar = self.run_inference(x)

        kl = self.kl(q_mean, q_logvar).sum(1)
        z = self.reparameterized_sample(q_mean, q_logvar)
        pcfg_params = self.get_pcfg_params(z)

        logPx, mem = self.inside(x, pcfg_params, x_hid)
        return logPx, kl, mem

    def run_inference(self, x):
        layered_state, state_mask = self.emb_enc(x)     # (b, N, d)
        state = layered_state[-1]
        pooled = masked_max(state, state_mask.bool().unsqueeze(-1), dim=1)     # (b, d)
        q_mean, q_logvar = self.infer_net(pooled).split(self.z_dim, dim=1)
        return state, q_mean, q_logvar

    def reparameterized_sample(self, q_mean, q_logvar):
        z = q_mean  # use q_mean to approximate during evaluation
        if self.training:
            noise = q_mean.new_zeros(q_mean.size()).normal_(0, 1)
            z = q_mean + (.5 * q_logvar).exp() * noise  # z = u + sigma * noise
        return z

    def get_pcfg_params(self, z):
        b = z.size()[0]     # the pcfg is dependent on z, which is differently sampled in the batch
        # Pi_S (batch, NT)
        root_emb = self.root_emb.expand(b, self.hidden_sz)
        root_emb = torch.cat([root_emb, z], -1)
        roots = self.root_mlp(root_emb).log_softmax(-1)

        # Pi_{P->w} (batch, T, V)
        term_emb = self.term_emb.unsqueeze(0).expand(b, self.T, self.hidden_sz)
        z_term_ex = z.unsqueeze(1).expand(b, self.T, self.z_dim)
        term_emb = torch.cat([term_emb, z_term_ex], -1)
        term_logp = self.term_mlp(term_emb).log_softmax(-1)

        # Pi_{A->BC} (batch, NT, NT+T, NT+T)
        nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(b, self.NT, self.hidden_sz)
        z_nonterm_ex = z.unsqueeze(1).expand(b, self.NT, self.z_dim)
        nonterm_emb = torch.cat([nonterm_emb, z_nonterm_ex], -1)
        rule_logp = self.rule_mlp(nonterm_emb).log_softmax(-1).reshape(b, self.NT, self.NT_T, self.NT_T)

        return roots, term_logp, rule_logp

    def inside(self, x, pcfg_params, x_hid=None):
        # x, x_mask: (batch, n)
        # x_hid: (batch, n, hid), must be of the same length as x
        # roots: (batch, NT)
        # terms: (batch, T, V)
        # rules: (batch, NT, NT+T, NT+T)
        roots, term_logp, rule_logp = pcfg_params
        b, n = x.size()
        score = x.new_full((b, n, n, self.NT_T), -1e9)  # the chart jointly saves nonterminals and preterminals
        NTs = slice(0, self.NT)                         # NTs proceed all Ts
        Ts = slice(self.NT, self.NT_T)
        un_ = torch.unsqueeze    # alias, for brevity
        tr_ = torch.arange

        # 0, (batch, n, T)
        level0 = un_(term_logp, 1).expand(b, n, self.T, self.V).gather(
            index=un_(un_(x, -1).expand(b, n, self.T), -1), dim=-1,
        ).squeeze(-1)
        score[:, 0, :, Ts] = level0

        emb_chart = None
        if x_hid is not None:
            emb_chart = x_hid.new_zeros(b, n, n, self.NT_T, self.encoding_dim)
            term_hid = self.unary_composer_term(self.term_emb)  # (T, hid)
            word_hid = self.unary_composer_word(x_hid)          # (b, n, hid)
            emb_chart[:, 0, :, Ts] = unit_norm(un_(un_(term_hid, 0), 1) + un_(word_hid, 2))   # (b, n, T, hid)

            a_hid = self.binary_composer_head(self.nonterm_emb)[None, None, :]  # (1, 1, NT, hid)

        device = x.device
        for width in range(1, n):
            coordinates = self._get_inside_coordinates(n, width, device)
            b_score, c_score = self._inside_chart_select(score, coordinates)
            # *_score: (batch, pos=(n - width), arrangement=width, NT_T, NT_T)
            b_score = b_score.unsqueeze(-1)
            c_score = c_score.unsqueeze(-2)

            # score_arr: (batch, pos=(n-width), A=NT, arrangement=width, B=NT_T, C=NT_T)
            score_arr = un_(b_score + c_score, 2) + un_(un_(rule_logp, 1), 3)
            # a_score: (batch, pos, NT)
            a_score = torch.logsumexp(score_arr, dim=(3, 4, 5))
            score[:, width, :n - width, NTs] = a_score

            if x_hid is not None:
                # to save storage, the computation is factored into 3 parts
                # recall score_arr: (batch, pos, A=NT, arrangement, B, C)

                # b/c_hid: (batch, pos, 1, arrangement, B/C, hid)
                b_repr, c_repr = self._inside_chart_select(emb_chart, coordinates)
                b_hid = self.binary_composer_left(b_repr).unsqueeze(2)
                c_hid = self.binary_composer_right(c_repr).unsqueeze(2)

                # b/c_weight: (batch, pos, A, arrangement, B/C, 1)
                b_weight = torch.logsumexp(score_arr, dim=(-1)).unsqueeze(-1)
                c_weight = torch.logsumexp(score_arr, dim=(-2)).unsqueeze(-1)

                # b/c_factor: (batch, pos, A, hid)
                b_factor = (b_hid * b_weight.exp()).sum(dim=(3, 4))
                c_factor = (c_hid * c_weight.exp()).sum(dim=(3, 4))

                # a_hid: (1, 1, A, hid)
                # a_score: (batch, pos, A)
                a_factor = a_hid * un_(a_score, -1).exp()

                normalized_abc = unit_norm(a_factor + b_factor + c_factor)
                emb_chart[:, width, :n - width, NTs] = normalized_abc

        lengths = (x != self.padding).sum(-1)
        logPxs = score[tr_(b, device=device), lengths - 1, 0, NTs] + roots       # (b, NT)
        logPx = torch.logsumexp(logPxs, dim=1)  # sum out the start NT
        if x_hid is not None:
            chart = emb_chart[:, 1:, :, NTs]     # (b, n - 1, n, NT, hid)
            mem = (roots.reshape(b, 1, 1, self.NT, 1).exp() * chart).sum(3).reshape(b, (n - 1) * n, -1)
            return logPx, mem
        return logPx

    def _get_inside_coordinates(self, n: int, width: int, device):
        un_ = torch.unsqueeze
        tr_ = torch.arange
        lvl_b = un_(tr_(width, device=device), 0)  # (pos=1, sublvl)
        pos_b = un_(tr_(n - width, device=device), 1)  # (pos, subpos=1)
        lvl_c = un_(tr_(width - 1, -1, -1, device=device), 0)  # (pos=1, sublvl), reversed lvl_b
        pos_c = un_(tr_(1, width + 1, device=device), 0) + pos_b  # (pos=(n-width), subpos=width))
        return lvl_b, pos_b, lvl_c, pos_c

    def _inside_chart_select(self, score_chart, coordinates):
        lvl_b, pos_b, lvl_c, pos_c = coordinates
        # *_score: (batch, pos=(n - width), arrangement=width, NT_T)
        b_score = score_chart[:, lvl_b, pos_b].clone()
        c_score = score_chart[:, lvl_c, pos_c].clone()
        return b_score, c_score

    @torch.no_grad()
    def generate(self, pcfg_params, max_steps=80):
        # roots: (b, NT)
        # terms: (b, T, V)
        # rules: (b, NT, NT_T, NT_T)
        roots, terms, rules = pcfg_params
        batch_sz, _, vocab_sz = terms.size()

        stack = TensorBatchStack(batch_sz, self.NT, 1, dtype=torch.long, device=roots.device)
        output = TensorBatchStack(batch_sz, self.NT, 1, dtype=torch.long, device=roots.device)

        succ = stack.push(roots.argmax(-1, keepdim=True), push_mask=None)
        step: int = 0
        stack_not_empty: torch.BoolTensor = stack.top()[1] > 0    # (batch,)
        while succ.bool().any() and stack_not_empty.any() and step < max_steps:
            # (batch, 1), (batch,)
            token, pop_succ = stack.pop(stack_not_empty * succ)   # only nonterm tokens stored on the stack
            succ *= pop_succ
            lhs_is_nt = (token < self.NT).squeeze()
            lhs_not_nt = ~lhs_is_nt

            words = self._generate_next_term(pcfg_params, token, lhs_not_nt * succ)
            output.push(words, push_mask=lhs_not_nt * succ)

            rhs_b, rhs_c = self._generate_next_nonterms(pcfg_params, token, lhs_is_nt * succ)
            succ *= stack.push(rhs_c, lhs_is_nt * succ)
            succ *= stack.push(rhs_b, lhs_is_nt * succ)

            stack_not_empty: torch.BoolTensor = stack.top()[1] > 0    # (batch,)
            step += 1

        buffer, mask = output.dump()
        return buffer, mask

    @staticmethod
    def kl(mean, logvar):
        # mean, logvar: (batch, z_dim)
        result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
        return result

    def _generate_next_term(self, pcfg_params, token, term_mask) -> torch.LongTensor:
        roots, terms, _ = pcfg_params
        batch_sz, _, vocab_sz = terms.size()
        batch_range = torch.arange(batch_sz, device=roots.device)
        words = terms[batch_range, (token - self.NT).squeeze() * term_mask].argmax(-1, keepdim=True)
        return words    # (batch, 1)

    def _generate_next_nonterms(self, pcfg_params, token, nonterm_mask) -> Tuple[torch.LongTensor, torch.LongTensor]:
        roots, terms, rules = pcfg_params
        batch_sz, _, vocab_sz = terms.size()
        rules = rules.reshape(batch_sz, self.NT, -1)    # (batch, NT, NT_T ** 2)

        batch_range = torch.arange(batch_sz, device=roots.device)
        rhs = rules[batch_range, token.squeeze() * nonterm_mask].argmax(1, keepdim=True)    # (batch, 1)

        rhs_b = rhs.div(self.NT_T, rounding_mode='floor')  # equivalent to py3 //
        rhs_c = rhs % self.NT_T

        return rhs_b, rhs_c     # (batch, 1), (batch, 1)

