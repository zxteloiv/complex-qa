from typing import Optional, Tuple, Literal, Union, List

import torch
import torch.nn as nn

from .base_pcfg import PCFGMixin, PCFGModule
from .res import ResLayer

from ..interfaces.encoder import EmbedAndEncode


def unit_norm(x, p=2, eps=1e-12):
    return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


def _un(x: torch.Tensor, dims: Union[List[int], int]):
    if isinstance(dims, int):
        return x.unsqueeze(dims)
    else:
        for d in sorted(dims):
            x = x.unsqueeze(d)
        return x


class CompoundPCFG(PCFGModule):
    def __init__(self,
                 num_nonterminal: int,
                 num_preterminal: int,
                 num_vocab_token: int,
                 hidden_sz: int,
                 encoder_input_dim: Optional[int],
                 z_dim: Optional[int] = None,
                 emb_chart_dim: int = None,
                 padding_id: int = 0,
                 ):
        super().__init__()
        #
        # For PCFG Params: (hidden_sz (embedding) + z_dim (external),) -> (start, binary, unary rules)
        # For PCFG as an encoder: (encoder_input_dim,) -> (encoding_dim,)
        #

        self.NT = num_nonterminal
        self.T = num_preterminal
        self.V = num_vocab_token
        self.padding = padding_id

        self.hidden_sz = hidden_sz
        self.z_dim = z_dim or hidden_sz
        self.emb_chart_dim = emb_chart_dim or hidden_sz
        self.encoder_input_dim = encoder_input_dim

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

        if encoder_input_dim is not None:   # otherwise no encoding is required, only for generation
            self._init_encoder_modules()

        self._initialize()

    def _init_encoder_modules(self):
        chart_dim = self.emb_chart_dim
        self.unary_word = nn.Sequential(nn.Linear(self.encoder_input_dim, chart_dim),
                                        ResLayer(chart_dim, chart_dim))
        # term embedding to chart dim
        self.unary_preterm = nn.Sequential(nn.Linear(self.hidden_sz, chart_dim),
                                           ResLayer(chart_dim, chart_dim))
        self.binary_head = nn.Sequential(nn.Linear(self.hidden_sz, chart_dim),
                                         ResLayer(chart_dim, chart_dim))
        self.binary_left = ResLayer(chart_dim, chart_dim)
        self.binary_right = ResLayer(chart_dim, chart_dim)

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

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
        device = x.device
        NTs = slice(0, self.NT)  # NTs proceed all Ts

        def tr_(*args, **kwargs):
            return torch.arange(*args, **kwargs, device=device)

        score = self._score_chart_init(x, pcfg_params)           # (b, n, n, NT_T)
        emb_chart = self._emb_chart_init(x_hid, pcfg_params)     # (b, n, n, NT_T, hid)
        # a_hid: (NT, hid)
        a_hid = self._get_category_embeddings()

        for width in range(1, n):
            coordinates = self.get_inside_coordinates(n, width, device)
            # b/c_score: (batch, pos=(n - width), arrangement=width, NT_T)
            # rule_logp: (b, A, B, C)
            b_score, c_score = self.inside_chart_select(score, coordinates)

            # (b, pos, ~~arr~~, B, C)
            bc_score = (_un(b_score, -1) + _un(c_score, -2)).logsumexp(dim=2, keepdim=True)

            # (b, pos, A, ~~B~~, ~~C~~) <-- (b, 1, A, B, C) + (b, pos, 1, B, C)
            a_score = (_un(rule_logp, 1) + bc_score).logsumexp(dim=(3, 4))
            score[:, width, :n - width, NTs] = a_score

            if x_hid is not None:
                chart_layer = self._get_emb_chart_layer(emb_chart, coordinates, pcfg_params,
                                                        b_score=b_score, c_score=c_score,
                                                        a_score=a_score, a_hid=a_hid,
                                                        )
                emb_chart[:, width, :n - width, NTs] = unit_norm(chart_layer)

        lengths = (x != self.padding).sum(-1)
        logPxs = score[tr_(b), lengths - 1, 0, NTs] + roots       # (b, NT)
        logPx = torch.logsumexp(logPxs, dim=1)  # sum out the start NT
        if x_hid is not None:
            # fixing the length layer embedding
            chart = emb_chart[:, 1:, :, NTs]     # (b, n - 1, n, A, hid)
            chart_rs = chart.reshape(b, (n - 1) * n, self.NT, self.emb_chart_dim)
            mem = (chart_rs * _un(roots, [1, 3]).exp()).sum(dim=2)  # (b, n(n-1), hid)
            return logPx, mem
        return logPx

    def _score_chart_init(self, x, pcfg_params):
        # x, x_mask: (batch, n)
        # x_hid: (batch, n, hid), must be of the same length as x
        # roots: (batch, NT)
        # terms: (batch, T, V)
        # rules: (batch, NT, NT+T, NT+T)
        roots, term_logp, rule_logp = pcfg_params
        b, n = x.size()
        s_chart = x.new_full((b, n, n, self.NT_T), -1e12)  # the chart jointly saves nonterminals and preterminals

        # 0, (batch, n, T, 1)
        level0 = _un(term_logp, 1).expand(b, n, self.T, self.V).gather(
            index=_un(_un(x, -1).expand(b, n, self.T), -1), dim=-1,
        )
        s_chart[:, 0, :, self.NT:self.NT_T] = level0.squeeze(-1)
        return s_chart

    def _emb_chart_init(self, x_hid, pcfg_params):
        if x_hid is None:
            return None

        b, n = x_hid.size()[:2]
        emb_chart = x_hid.new_zeros(b, n, n, self.NT_T, self.emb_chart_dim)
        word = self.unary_word(x_hid)                   # (b, n,    hid)
        preterm = self.unary_preterm(self.term_emb)     #       (P, hid)
        emb_chart[:, 0, :, self.NT:self.NT_T] = unit_norm(_un(word, 2) + _un(preterm, [0, 1]))
        return emb_chart

    def _get_category_embeddings(self):
        a_hid = self.binary_head(self.nonterm_emb)          # (NT, hid)
        return a_hid

    def _get_emb_chart_layer(self, emb_chart, coordinates, pcfg_params, **ctx):
        # a_hid: (A=NT, hid)
        a_hid = ctx['a_hid']
        # a_score: (b, pos, A), b/c_score: (b, pos, arr, B/C)
        a_score, b_score, c_score = ctx['a_score'], ctx['b_score'], ctx['c_score']
        # roots: (batch, A)
        # terms: (batch, T, V)
        # rules: (batch, A, B, C)
        roots, terms, rules = pcfg_params

        # l/r_cell, b/c_hid: (b, pos, arr, B/C=NT_T, hid)
        l_cell, r_cell = self.inside_chart_select(emb_chart, coordinates)
        b_hid = self.binary_left(l_cell)
        c_hid = self.binary_right(r_cell)

        # (b, pos, arr, 1, B/C, hid)
        b_factor = (b_hid * _un(b_score, -1).exp()).unsqueeze(3)
        c_factor = (c_hid * _un(c_score, -1).exp()).unsqueeze(3)

        # (b, pos, arr, A, B, 1) <--
        #          (b, 1, 1, A, B, C)  + (b, pos, arr, 1, 1, C)
        b_weight = (_un(rules, [1, 2]) + _un(c_score, [3, 4])).logsumexp(dim=5, keepdim=True).exp()

        # (b, pos, arr, A, C, 1) <-- (note the difference of sum on B@dim4, instead of C@dim5 above
        #          (b, 1, 1, A, B, C)  + (b, pos, arr, 1, B, 1)
        c_weight = (_un(rules, [1, 2]) + _un(b_score, [3, 5])).logsumexp(dim=4).unsqueeze(-1).exp()

        # (b, pos, ~~arr~~, A, ~~B/C~~, hid)
        b_item = (b_factor * b_weight).sum(dim=(2, 4))
        c_item = (c_factor * c_weight).sum(dim=(2, 4))

        # (b, pos, A, 1)
        a_score_ex = _un(a_score, -1).exp()

        # (b, pos, A, hid)
        chart_layer = (b_item + c_item) / (a_score_ex + 1e-30) + _un(a_hid, [0, 1])
        return chart_layer

    def generate_next_term(self, pcfg_params, token, term_mask) -> torch.LongTensor:
        roots, terms, _ = pcfg_params
        batch_sz, _, vocab_sz = terms.size()
        batch_range = torch.arange(batch_sz, device=roots.device)
        words = terms[batch_range, (token - self.NT).squeeze() * term_mask].argmax(-1, keepdim=True)
        return words    # (batch, 1)

    def generate_next_nonterms(self, pcfg_params, token, nonterm_mask) -> Tuple[torch.LongTensor, torch.LongTensor]:
        roots, terms, rules = pcfg_params
        batch_sz, _, vocab_sz = terms.size()
        rules = rules.reshape(batch_sz, self.NT, -1)    # (batch, NT, NT_T ** 2)

        batch_range = torch.arange(batch_sz, device=roots.device)
        rhs = rules[batch_range, token.squeeze() * nonterm_mask].argmax(1, keepdim=True)    # (batch, 1)

        rhs_b = rhs.div(self.NT_T, rounding_mode='floor')  # equivalent to py3 //
        rhs_c = rhs % self.NT_T

        return rhs_b, rhs_c     # (batch, 1), (batch, 1)

    def get_encoder_output_size(self):
        return self.emb_chart_dim
