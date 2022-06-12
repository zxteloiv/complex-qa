import torch
from torch import nn
from .C_PCFG import CompoundPCFG, _un, unit_norm
from .res import ResLayer


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

























