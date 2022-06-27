from typing import Optional, List, Union, Tuple, Literal

import torch
import torch.nn as nn
from torch import nn as nn

from .fn import unit_norm, _un
from .base_pcfg import PCFGModule
from .res import ResLayer


class TNPCFG(PCFGModule):
    def __init__(self,
                 rank: int,
                 num_nonterminal: int,
                 num_preterminal: int,
                 num_vocab_token: int,
                 hidden_sz: int,
                 encoder_input_dim: int = None,
                 z_dim: int = 0,    # z_dim=0 means no compound inference required
                 emb_chart_dim: int = None,
                 padding_id: int = 0,
                 ):
        super().__init__()

        self.r = rank

        self.NT = num_nonterminal
        self.T = num_preterminal
        self.V = num_vocab_token
        self.NT_T = self.NT + self.T
        self.padding = padding_id

        self.hidden_sz = hidden_sz
        self.z_dim = z_dim or hidden_sz
        self.emb_chart_dim = emb_chart_dim or hidden_sz
        self.encoder_input_dim = encoder_input_dim

        self.term_emb = nn.Parameter(torch.randn(self.T, self.hidden_sz))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.hidden_sz))
        self.root_emb = nn.Parameter(torch.randn(1, self.hidden_sz))

        self.root_mlp = nn.Sequential(nn.Linear(self.hidden_sz + self.z_dim, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      nn.Linear(self.hidden_sz, self.NT))

        self.term_mlp = nn.Sequential(nn.Linear(self.hidden_sz + self.z_dim, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      nn.Linear(self.hidden_sz, self.V))

        self.parent_mlp, self.left_mlp, self.right_mlp = (
            nn.Sequential(nn.Linear(self.hidden_sz + self.z_dim, self.hidden_sz),
                          ResLayer(self.hidden_sz, self.hidden_sz),
                          ResLayer(self.hidden_sz, self.hidden_sz),
                          nn.Linear(self.hidden_sz, self.r))
            for _ in range(3)
        )

        if encoder_input_dim is not None:   # otherwise no encoding is required, only for generation
            self._init_encoder_modules()
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _init_encoder_modules(self):
        # compose embeddings according to unary and binary rules
        chart_dim = self.emb_chart_dim
        self.unary_word = nn.Sequential(nn.Linear(self.encoder_input_dim, chart_dim),
                                        ResLayer(chart_dim, chart_dim))

        self.unary_term = nn.Sequential(nn.Linear(self.hidden_sz, chart_dim),
                                        ResLayer(chart_dim, chart_dim))

        self.binary_left = ResLayer(chart_dim, chart_dim)
        self.binary_right = ResLayer(chart_dim, chart_dim)
        self.binary_head = nn.Sequential(nn.Linear(self.hidden_sz, chart_dim),
                                         ResLayer(chart_dim, chart_dim))

    def get_pcfg_params(self, z):
        b = z.size()[0]

        # -----------------------
        # roots: (b, NT)
        root_emb = self.root_emb.expand(b, self.hidden_sz)
        root_emb = torch.cat([root_emb, z], -1)
        roots = self.root_mlp(root_emb).log_softmax(-1)

        # -----------------------
        # unary rules: P -> x
        # term: (b, T, V)
        z_term_ex = z.unsqueeze(1).expand(b, self.T, self.z_dim)
        term_emb = self.term_emb.unsqueeze(0).expand(b, self.T, self.hidden_sz)
        term_cat = torch.cat([term_emb, z_term_ex], -1)
        terms = self.term_mlp(term_cat).log_softmax(-1)

        # -----------------------
        # decomposed binary rules: A -> B C
        z_nonterm_ex = z.unsqueeze(1).expand(b, self.NT, self.z_dim)
        nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(b, self.NT, self.hidden_sz)
        nonterm_cat = torch.cat([nonterm_emb, z_nonterm_ex], dim=-1)

        # head: (b, NT, r)
        head = self.parent_mlp(nonterm_cat).log_softmax(-1)

        # left/right: (b, NT_T, r)
        rule_cat = torch.cat([nonterm_cat, term_cat], dim=1)
        left = self.left_mlp(rule_cat).log_softmax(-2)
        right = self.right_mlp(rule_cat).log_softmax(-2)

        return roots, terms, (head, left, right)

    def inside(self, x, pcfg_params, x_hid=None):
        # x, x_mask: (batch, n)
        # x_hid: (batch, n, hid), must be of the same length as x
        logPx, s_chart = self._inside_for_score(x, pcfg_params)
        if x_hid is not None:
            emb_chart = self._inside_for_encoder(x_hid, x != self.padding, pcfg_params, s_chart)
            mem = self._get_final_attention_memory(emb_chart, s_chart, pcfg_params)
            return logPx, mem
        return logPx

    def _inside_for_score(self, x, pcfg_params):
        # roots: (batch, NT)
        # term: (batch, T, r), normalized on dim=-1
        # word: (batch, V, r), normalized on dim=-2
        # head: (batch, NT, r), normalized on dim=-1
        # left/right: (batch, NT_T, r), normalized on dim=-2
        roots, terms, (head, left, right) = pcfg_params
        b, n = x.size()
        NTs = slice(0, self.NT)                         # NTs proceed all Ts

        score = self._score_chart_init(x, pcfg_params)              # (b, n, n, NT_T)
        device = x.device
        for width in range(1, n):
            coordinates = self.get_inside_coordinates(n, width, device)
            # b/c_score: (batch, pos=(n - width), arrangement=width, NT_T)
            b_score, c_score = self.inside_chart_select(score, coordinates)

            # left/right: (batch, 1, 1, NT_T, r) <- (batch, NT_T, r)
            # vb/wc_score: (batch, pos, arr, r)
            vb_score = (_un(left, [1, 2]) + _un(b_score, -1)).logsumexp(dim=-2)
            wc_score = (_un(right, [1, 2]) + _un(c_score, -1)).logsumexp(dim=-2)

            vbwc_score = vb_score + wc_score    # (b, pos, arr, r)
            uvbwc_score = _un(head, 1) + vbwc_score.logsumexp(dim=2, keepdim=True)  # (b, pos, NT, r)
            a_score = uvbwc_score.logsumexp(-1)                                     # (b, pos, NT)
            score[:, width, :n - width, NTs] = a_score

        lengths = (x != self.padding).sum(-1)
        logPxs = score[torch.arange(b, device=device), lengths - 1, 0, NTs] + roots       # (b, NT)
        logPx = torch.logsumexp(logPxs, dim=1)  # sum out the start nonterm S
        return logPx, score

    def _score_chart_init(self, x, pcfg_params):
        # x, x_mask: (batch, n)
        # roots: (batch, NT)
        # terms: (batch, T, V)
        # head: (batch, NT, r), normalized on dim=-1
        # left/right: (batch, NT_T, r), normalized on dim=-2
        roots, terms, (head, left, right) = pcfg_params
        b, n = x.size()
        s_chart = x.new_full((b, n, n, self.NT_T), -1e12)

        level0 = _un(terms, 1).expand(b, n, self.T, self.V).gather(
            index=_un(_un(x, -1).expand(b, n, self.T), -1), dim=-1,
        )
        s_chart[:, 0, :, self.NT:] = level0.squeeze(-1)
        return s_chart

    def _inside_for_encoder(self, x_hid, x_mask, pcfg_params, s_chart):
        roots, terms, (head, left, right) = pcfg_params
        b, n = x_hid.size()[:2]
        a_hid = _un(self.binary_head(self.nonterm_emb), [0, 1])  # (1, 1, A, hid)
        emb_chart = self._emb_chart_init(x_hid, pcfg_params)     # (b, n, n, r, hid) or None
        for width in range(1, n):
            coordinates = self.get_inside_coordinates(n, width, x_hid.device)
            # b/c_score: (batch, pos=(n - width), arrangement=width, NT_T)
            b_score, c_score = self.inside_chart_select(s_chart, coordinates)

            # left/right -> (b, 1, 1, B/C=NT_T, r)
            # b/c_score -> (b, pos, arr, B/C, 1)
            # vb/wc_score: (b, pos, arr, B/C, r)
            vb_score = (_un(left, [1, 2]) + _un(b_score, -1)).exp()
            wc_score = (_un(right, [1, 2]) + _un(c_score, -1)).exp()

            # b/c_hid: (b, pos, arr, B/C, hid)
            b_repr, c_repr = self.inside_chart_select(emb_chart, coordinates)
            b_hid = self.binary_left(b_repr)
            c_hid = self.binary_right(c_repr)

            # b_rank: (b, pos, arr, ~~B~~, r, hid)
            # b_weight: (b, pos, arr, ~~C~~, r, 1)
            b_rank = (_un(b_hid, -2) + _un(vb_score, -1)).sum(dim=3)
            b_weight = wc_score.sum(dim=-2).unsqueeze(-1)
            # b_factor: (b, pos, ~~arr~~, r, hid)
            b_factor = (b_weight * b_rank).sum(dim=2)

            # (b, pos, arr, ~~C~~, r, hid) <-- (b, pos, arr, C, 1, hid) + (b, pos, arr, C, r, 1)
            c_rank = (_un(c_hid, -2) + _un(wc_score, -1)).sum(dim=3)
            # c_weight: (b, pos, arr, ~~B~~, r, 1)
            c_weight = vb_score.sum(dim=-2).unsqueeze(-1)
            # c_factor: (b, pos, ~~arr~~, r, hid)
            c_factor = (c_weight * c_rank).sum(dim=2)

            # (b, pos, A, ~~r~~, hid)
            bc_factor = (_un(b_factor + c_factor, 2) * _un(head.exp(), [1, 4])).sum(dim=-2)

            a_score = s_chart[:, width, :n - width, :self.NT].unsqueeze(-1)     # (b, pos, A, 1)

            layer = bc_factor / (a_score.exp() + 1e-31) + a_hid
            emb_chart[:, width, :n - width, :self.NT] = unit_norm(layer)

        return emb_chart

    def _emb_chart_init(self, x_hid, pcfg_params):
        b, n = x_hid.size()[:2]
        emb_chart = x_hid.new_zeros(b, n, n, self.NT_T, self.emb_chart_dim)

        # x_hid: (batch, n, hid)
        x_unary = _un(self.unary_word(x_hid), 2)                 # (b, n, 1, hid)
        t_unary = _un(self.unary_term(self.term_emb), [0, 1])    # (1, 1, T, hid)
        emb_chart[:, 0, :, self.NT:] = unit_norm(x_unary + t_unary)     # (b, n, T, hid)
        return emb_chart

    def _get_final_attention_memory(self, emb_chart, s_chart, pcfg_params):
        b, n = emb_chart.size()[:2]
        # roots: (b, A)
        roots, terms, (head, left, right) = pcfg_params

        # (b, n-1, n, A, hid)
        chart = emb_chart[:, 1:, :, :self.NT]
        # (b, n(n-1), A, hid)
        mem_a = chart.reshape(b, (n - 1) * n, self.NT, self.emb_chart_dim)

        mem = (_un(roots, [1, 3]).exp() * mem_a).sum(dim=2)
        return mem

    def _generate_next_nonterms(self,
                                pcfg_params,    # (b, NT), (b, T, V), [(b, NT, r), (b, NT_T, r) * 2]
                                token,          # (b, 1)
                                nonterm_mask    # (b,)
                                ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        roots, terms, (head, left, right) = pcfg_params
        batch_sz = head.size()[0]
        batch_range = torch.arange(batch_sz, device=head.device, dtype=torch.long)
        head_r = head[batch_range, token.squeeze() * nonterm_mask].unsqueeze(1)     # (batch, 1, r)
        rhs_b = torch.logsumexp(head_r + left, dim=-1).argmax(-1, keepdim=True)     # (batch, 1)
        rhs_c = torch.logsumexp(head_r + right, dim=-1).argmax(-1, keepdim=True)    # (batch, 1)
        return rhs_b, rhs_c

    def get_encoder_output_size(self):
        return self.emb_chart_dim


class ReducedTDPCFG(TNPCFG):
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

    def _inside_for_encoder(self, x_hid, x_mask, pcfg_params, s_chart):
        # roots: (b, A), head: (b, A, r), left/right: (b, B/C, r), terms: (b, T, V)
        roots, terms, (head, left, right) = pcfg_params
        b, n = x_hid.size()[:2]
        a_hid, b_hid, c_hid = self._get_category_embeddings()   # (A, hid), (B, hid), (C, hid)
        emb_chart = self._emb_chart_init(x_hid, pcfg_params)    # (b, n, n, hid)
        for width in range(1, n):
            coordinates = self.get_inside_coordinates(n, width, x_hid.device)
            b_score, c_score = self.inside_chart_select(s_chart, coordinates)   # (b, pos, arr, B/C)

            r_score_arr = head + _un(roots, -1)        # (b, A, r)
            r_score = r_score_arr.logsumexp(dim=1)     # (b, r)
            a_item = (_un(r_score_arr, -1).exp() * _un(a_hid, [0, 2])).sum(dim=1)     # (b, ~~A~~, r, hid)

            # vb/wc_score_arr: (b, pos, arr, B/C, r)
            vb_score_arr = _un(left, [1, 2]) + _un(b_score, -1)
            wc_score_arr = _un(right, [1, 2]) + _un(c_score, -1)

            # vb/wc_score: (b, pos, arr, r)
            vb_score = vb_score_arr.logsumexp(dim=-2)
            wc_score = wc_score_arr.logsumexp(dim=-2)
            vbwc_score = vb_score + wc_score

            # b/c_item: (b, pos, arr, r, hid) <-- (b, pos, arr, B/C, r, hid)
            b_item = (_un(vb_score_arr, -1).exp() * _un(b_hid, [0, 1, 2, 4])).sum(dim=3)
            c_item = (_un(wc_score_arr, -1).exp() * _un(c_hid, [0, 1, 2, 4])).sum(dim=3)

            # cell_item: (b, pos, arr, hid)
            l_cell, r_cell = self.inside_chart_select(emb_chart, coordinates)
            cell_item = self.binary_left(l_cell) + self.binary_right(r_cell)

            # cell_factor: (b, pos, r, hid) <-- (b, pos, arr, r, hid)
            cell_factor = (_un(cell_item, -2) * _un(vbwc_score, -1).exp()).sum(dim=2)
            # b_factor: (b, pos, r, hid) <-- (b, pos, arr, r, hid)
            b_factor = (b_item * _un(wc_score, -1).exp()).sum(dim=2)
            # c_factor: (b, pos, r, hid) <-- (b, pos, arr, r, hid)
            c_factor = (c_item * _un(vb_score, -1).exp()).sum(dim=2)

            # cell_b_c: (b, pos, hid) <-- (b, pos, r, hid)
            cell_b_c = (_un(r_score, [1, 3]).exp() * (cell_factor + b_factor + c_factor)).sum(dim=2)

            # a_factor: (b, pos, ~~r~~, hid)
            a_factor = (_un(a_item, 1) * _un(vbwc_score.exp().sum(dim=2), -1)).sum(dim=2)

            # (b, pos, hid)
            span_hid = a_factor + cell_b_c

            # denominator: (b, pos, 1) <-- (b, pos, A)
            a_score = s_chart[:, width, :n - width, :self.NT]   # (b, pos, A)
            denominator = (a_score + _un(roots, 1)).exp().sum(dim=-1, keepdim=True).clamp(min=1e-12)

            chart_layer = span_hid / denominator    # (b, pos, hid)
            emb_chart[:, width, :n - width] = unit_norm(chart_layer)
        return emb_chart

    def _emb_chart_init(self, x_hid, pcfg_params):
        if x_hid is None:
            return None

        b, n = x_hid.size()[:2]
        emb_chart = x_hid.new_zeros(b, n, n, self.emb_chart_dim)
        word = self.unary_word(x_hid)  # (b, n,    hid)
        emb_chart[:, 0] = unit_norm(word)
        return emb_chart

    def _get_category_embeddings(self):
        a_hid = self.binary_head(self.nonterm_emb)
        cat_emb = torch.cat([self.nonterm_emb, self.term_emb], dim=0)
        b_hid = self.binary_left_symbol(cat_emb)
        c_hid = self.binary_right_symbol(cat_emb)
        return a_hid, b_hid, c_hid  # (A, hid), (B, hid), (C, hid)

    def _get_final_attention_memory(self, emb_chart, s_chart, pcfg_params):
        b, n = emb_chart.size()[:2]
        # fixing the length layer embedding
        chart = emb_chart[:, 1:]  # (b, n - 1, n, hid)
        chart_rs = chart.reshape(b, (n - 1) * n, self.emb_chart_dim)
        return chart_rs
