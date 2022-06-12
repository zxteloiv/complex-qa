import torch
import torch.nn as nn

from .C_PCFG import _un
from .TN_PCFG import TNPCFG, ResLayer, unit_norm


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
