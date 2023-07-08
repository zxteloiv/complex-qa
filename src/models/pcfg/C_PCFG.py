from typing import Optional, Tuple

import torch
from torch import nn

from .base_pcfg import PCFGModule
from .fn import unit_norm, _un
from models.pcfg.res import ResLayer


class CompoundPCFG(PCFGModule):
    def __init__(self,
                 num_nonterminal: int,
                 num_preterminal: int,
                 num_vocab_token: int,
                 hidden_sz: int,
                 encoder_input_dim: Optional[int],
                 z_dim: int = 0,    # z_dim=0 means no compound inference required
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
        cat_hid = self._get_category_embeddings()

        for width in range(1, n):
            coordinates = self.get_inside_coordinates(n, width, device)
            # b/c_score: (batch, pos=(n - width), arrangement=width, NT_T)
            # rule_logp: (b, A, B, C)
            b_score, c_score = self.inside_chart_select(score, coordinates)

            # (b, pos, A, arr, B, C)
            score_arr = _un(_un(b_score, -1) + _un(c_score, -2), 2) + _un(rule_logp, [1, 3])
            # (b, pos, A)
            a_score = score_arr.logsumexp(dim=(3, 4, 5))
            score[:, width, :n - width, NTs] = a_score

            if x_hid is not None:
                chart_layer = self._get_emb_chart_layer(emb_chart, coordinates, pcfg_params,
                                                        score_arr=score_arr, a_score=a_score,
                                                        cat_hid=cat_hid,)
                self._set_emb_chart_layer(emb_chart, n, width, chart_layer)

        lengths = (x != self.padding).sum(-1)
        logPxs = score[tr_(b), lengths - 1, 0, NTs] + roots       # (b, NT)
        logPx = torch.logsumexp(logPxs, dim=1)  # sum out the start NT
        if x_hid is not None:
            mem = self._get_attention_memory(emb_chart, pcfg_params)
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
        a_hid = ctx['cat_hid']
        # a_score: (b, pos, A), score_arr: (b, pos, A, arr, B, C)
        a_score, score_arr = ctx['a_score'], ctx['score_arr']

        # l/r_cell: (b, pos, arr, B/C=NT_T, hid)
        l_cell, r_cell = self.inside_chart_select(emb_chart, coordinates)

        # b/c_hid: (b, pos, 1, arr, B/C=NT_T, hid)
        b_hid = self.binary_left(l_cell).unsqueeze(2)
        c_hid = self.binary_right(r_cell).unsqueeze(2)

        # (b, pos, A, arr, B/C, 1)
        b_weight = score_arr.logsumexp(dim=-1, keepdim=True)
        c_weight = score_arr.logsumexp(dim=-2).unsqueeze(-1)

        # (b, pos, A, ~~arr~~, ~~B/C~~, hid)
        b_factor = (b_weight.exp() * b_hid).sum(dim=(3, 4))
        c_factor = (c_weight.exp() * c_hid).sum(dim=(3, 4))

        # (b, pos, A, 1)
        a_score_ex = _un(a_score, -1).exp()

        # (b, pos, A, hid)
        chart_layer = (b_factor + c_factor) / (a_score_ex + 1e-30) + _un(a_hid, [0, 1])
        return chart_layer

    def _set_emb_chart_layer(self, emb_chart, n, width, chart_layer):
        emb_chart[:, width, :n - width, :self.NT] = unit_norm(chart_layer)

    def _get_attention_memory(self, emb_chart, pcfg_params):
        b, n = emb_chart.size()[:2]
        roots = pcfg_params[0]
        # fixing the length layer embedding
        chart = emb_chart[:, 1:, :, :self.NT]  # (b, n - 1, n, A, hid)
        chart_rs = chart.reshape(b, (n - 1) * n, self.NT, self.emb_chart_dim)
        mem = (chart_rs * _un(roots, [1, 3]).exp()).sum(dim=2)  # (b, n(n-1), hid)
        return mem

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


class ReducedCPCFG(CompoundPCFG):
    def _init_encoder_modules(self):
        chart_dim = self.emb_chart_dim
        self.unary_word = nn.Sequential(nn.Linear(self.encoder_input_dim, chart_dim),
                                        ResLayer(chart_dim, chart_dim))
        self.binary_head = nn.Sequential(nn.Linear(self.hidden_sz, chart_dim),
                                         ResLayer(chart_dim, chart_dim))
        self.binary_left_symbol = nn.Sequential(nn.Linear(self.hidden_sz, chart_dim),
                                                ResLayer(chart_dim, chart_dim))
        self.binary_right_symbol = nn.Sequential(nn.Linear(self.hidden_sz, chart_dim),
                                                ResLayer(chart_dim, chart_dim))
        self.binary_left = ResLayer(chart_dim, chart_dim)
        self.binary_right = ResLayer(chart_dim, chart_dim)

    def _emb_chart_init(self, x_hid, pcfg_params):
        if x_hid is None:
            return None

        b, n = x_hid.size()[:2]
        emb_chart = x_hid.new_zeros(b, n, n, self.emb_chart_dim)
        word = self.unary_word(x_hid)                   # (b, n,    hid)
        emb_chart[:, 0] = unit_norm(word)
        return emb_chart

    def _get_category_embeddings(self):
        a_hid = self.binary_head(self.nonterm_emb)
        cat_emb = torch.cat([self.nonterm_emb, self.term_emb], dim=0)
        b_hid = self.binary_left_symbol(cat_emb)
        c_hid = self.binary_right_symbol(cat_emb)
        return a_hid, b_hid, c_hid  # (A, hid), (B, hid), (C, hid)

    def _get_emb_chart_layer(self, emb_chart, coordinates, pcfg_params, **ctx):
        a_hid, b_hid, c_hid = ctx['cat_hid']
        roots, terms, rules = pcfg_params   # (b, A), (b, T, V), (b, A, B, C)
        a_score = ctx['a_score']            # (b, pos, A)
        score_arr = ctx['score_arr']        # (0b, 1pos, 2A, 3arr, 4B, 5C)

        #  0   1   2   3   4  5  6
        # (b, pos, A, arr, B, C, 1)
        full_score = _un((score_arr + _un(roots, [1, 3, 4, 5])).exp(), -1)

        # cell_factor: (b, pos, ~~arr~~, hid)
        l_cell, r_cell = self.inside_chart_select(emb_chart, coordinates)
        cell_hid = self.binary_left(l_cell) + self.binary_right(r_cell)
        cell_factor = (cell_hid * full_score.sum(dim=(2, 4, 5))).sum(dim=2)

        # b_factor: (b, pos, ~~B~~, hid)
        b_factor = (_un(b_hid, [0, 1]) * full_score.sum(dim=(2, 3, 5))).sum(dim=2)

        # c_factor: (b, pos, ~~C~~, hid)
        c_factor = (_un(c_hid, [0, 1]) * full_score.sum(dim=(2, 3, 4))).sum(dim=2)

        # a_factor: (b, pos, ~~A~~, hid)
        a_factor = (_un(a_hid, [0, 1]) * full_score.sum(dim=(3, 4, 5))).sum(dim=2)

        # (b, pos, 1) <-- (b, pos, A)
        denominator = (a_score + _un(roots, 1)).exp().sum(dim=-1, keepdim=True).clamp(min=1e-12)

        chart_layer = (a_factor + b_factor + c_factor + cell_factor) / denominator
        return chart_layer  # (b, pos, hid)

    def _set_emb_chart_layer(self, emb_chart, n, width, chart_layer):
        emb_chart[:, width, :n - width] = unit_norm(chart_layer)

    def _get_attention_memory(self, emb_chart, pcfg_params):
        b, n = emb_chart.size()[:2]
        # fixing the length layer embedding
        chart = emb_chart[:, 1:]  # (b, n - 1, n, hid)
        chart_rs = chart.reshape(b, (n - 1) * n, self.emb_chart_dim)
        return chart_rs
