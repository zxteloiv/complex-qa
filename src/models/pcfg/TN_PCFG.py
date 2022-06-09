from typing import Optional, List, Union, Tuple, Literal

import torch
import torch.nn as nn

from .C_PCFG import unit_norm, _un
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
                 z_dim: Optional[int] = None,
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

        self.term_emb = nn.Parameter(torch.randn(self.T, self.hidden_sz))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, self.hidden_sz))
        self.root_emb = nn.Parameter(torch.randn(1, self.hidden_sz))
        self.word_emb = nn.Parameter(torch.randn(self.V, self.hidden_sz))

        self.root_mlp = nn.Sequential(nn.Linear(self.hidden_sz + self.z_dim, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      ResLayer(self.hidden_sz, self.hidden_sz),
                                      nn.Linear(self.hidden_sz, self.NT))

        self.parent_mlp, self.left_mlp, self.right_mlp, self.term_mlp, self.word_mlp = (
            nn.Sequential(nn.Linear(self.hidden_sz + self.z_dim, self.hidden_sz),
                          ResLayer(self.hidden_sz, self.hidden_sz),
                          ResLayer(self.hidden_sz, self.hidden_sz),
                          nn.Linear(self.hidden_sz, self.r))
            for _ in range(5)
        )

        if encoder_input_dim is not None:   # otherwise no encoding is required, only for generation
            # compose embeddings according to unary and binary rules
            _hid_dim = emb_chart_dim
            self.unary_word = nn.Sequential(nn.Linear(encoder_input_dim, _hid_dim),
                                            ResLayer(_hid_dim, _hid_dim))

            self.unary_term = nn.Sequential(nn.Linear(hidden_sz, _hid_dim),
                                            ResLayer(_hid_dim, _hid_dim))

            self.binary_left = ResLayer(_hid_dim, _hid_dim)
            self.binary_right = ResLayer(_hid_dim, _hid_dim)
            self.binary_head = nn.Sequential(nn.Linear(hidden_sz, _hid_dim),
                                             ResLayer(_hid_dim, _hid_dim))
        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def get_pcfg_params(self, z):
        b = z.size()[0]

        # roots: (b, NT)
        root_emb = self.root_emb.expand(b, self.hidden_sz)
        root_emb = torch.cat([root_emb, z], -1)
        roots = self.root_mlp(root_emb).log_softmax(-1)

        # -----------------------
        # decomposed unary rules: P -> x, different from the original TD-PCFG paper
        # term: (b, T, r)
        z_term_ex = z.unsqueeze(1).expand(b, self.T, self.z_dim)
        term_emb = self.term_emb.unsqueeze(0).expand(b, self.T, self.hidden_sz)
        term_cat = torch.cat([term_emb, z_term_ex], -1)
        term = self.term_mlp(term_cat).log_softmax(-1)

        # word: (b, V, r)
        z_vocab_ex = z.unsqueeze(1).expand(b, self.V, self.z_dim)
        word_emb = self.word_emb.unsqueeze(0).expand(b, self.V, self.hidden_sz)
        word_cat = torch.cat([word_emb, z_vocab_ex], -1)
        word = self.word_mlp(word_cat).log_softmax(-2)

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

        return roots, (term, word), (head, left, right)

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
        roots, (term, word), (head, left, right) = pcfg_params
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
        # term: (batch, T, r)
        # word: (batch, V, r)
        # head: (batch, NT, r), normalized on dim=-1
        # left/right: (batch, NT_T, r), normalized on dim=-2
        roots, (term, word), (head, left, right) = pcfg_params
        b, n = x.size()
        s_chart = x.new_full((b, n, n, self.NT_T), -1e12)

        # term_logp: (b, T, V, ~~r~~)
        term_logp = (_un(term, 2) + _un(word, 1)).logsumexp(-1)
        # 0, (batch, n, T, 1)
        level0 = _un(term_logp, 1).expand(b, n, self.T, self.V).gather(
            index=_un(_un(x, -1).expand(b, n, self.T), -1), dim=-1,
        )
        s_chart[:, 0, :, self.NT:self.NT_T] = level0.squeeze(-1)
        return s_chart

    def _inside_for_encoder(self, x_hid, x_mask, pcfg_params, s_chart):
        roots, (term, word), (head, left, right) = pcfg_params
        b, n = x_hid.size()[:2]
        emb_chart = self._emb_chart_init(x_hid, pcfg_params)     # (b, n, n, r, hid) or None
        for width in range(1, n):
            coordinates = self.get_inside_coordinates(n, width, x_hid.device)
            # b/c_score: (batch, pos=(n - width), arrangement=width, NT_T)
            b_score, c_score = self.inside_chart_select(s_chart, coordinates)

            # left/right -> (b, 1, 1, NT_T, r)
            # b/c_score -> (b, pos, arr, NT_T, 1)
            # vb/wc_score: (b, pos, arr, ~~NT_T~~, r)
            vb_score = (_un(left, [1, 2]) + _un(b_score, -1)).logsumexp(dim=-2)
            wc_score = (_un(right, [1, 2]) + _un(c_score, -1)).logsumexp(dim=-2)

            vbwc_score = vb_score + wc_score    # (b, pos, arr, r)
            layer = self._get_emb_chart_layer(emb_chart, coordinates, pcfg_params, vbwc_score=vbwc_score)
            emb_chart[:, width, :n - width] = layer

        return emb_chart

    def _emb_chart_init(self, x_hid, pcfg_params):
        b, n = x_hid.size()[:2]
        emb_chart = x_hid.new_zeros(b, n, n, self.r, self.emb_chart_dim)

        roots, (term, word), (head, left, right) = pcfg_params

        # x_hid: (batch, n, hid)
        x_unary = self.unary_word(x_hid).unsqueeze(2)       # (b, n, 1, hid)

        t_unary = _un(self.unary_term(self.term_emb), 0)    # (1, T, hid)
        term_t = term.transpose(1, 2).exp()                 # (b, r, T)
        r_unary = _un(torch.matmul(term_t, t_unary), 1)     # (b, 1, r, hid)

        emb_chart[:, 0] = x_unary + r_unary                 # (b, n, r, hid)
        return emb_chart

    def _get_emb_chart_layer(self, emb_chart, coordinates, pcfg_params, **ctx):
        # to save storage, the computation is factored into 3 parts
        # recall score_arr: (batch, pos, A=NT, arrangement, B, C)
        vbwc_score = ctx['vbwc_score']      # (b, pos, arr, r)
        vbwc = _un(vbwc_score, -1).exp()    # (b, pos, arr, r, 1)

        # bc_hid: (b, pos, arr, r, hid)
        b_repr, c_repr = self.inside_chart_select(emb_chart, coordinates)
        bc_hid = self.binary_left(b_repr) + self.binary_right(c_repr)

        layer = (bc_hid * vbwc).sum(2)  # (b, pos, ~~arr~~, r, hid)
        return layer    # (b, pos, r, hid)

    def _get_final_attention_memory(self, emb_chart, s_chart, pcfg_params):
        b, n = emb_chart.size()[:2]
        roots, (term, word), (head, left, right) = pcfg_params

        chart = emb_chart[:, 1:]  # (b, n - 1, n, r, hid)
        chart_ex = chart.reshape(b, (n - 1) * n, self.r, self.emb_chart_dim)
        r_dist = (head + _un(roots, -1)).logsumexp(dim=-2)  # (b, r)
        mem_bc = (_un(r_dist, [1, 3]).exp() * chart_ex).sum(dim=2)  # (b, n(n-1), ~~r~~, hid)

        a_hid = _un(self.binary_head(self.nonterm_emb), [0, 1])  # (1, 1, NT, hid)
        a_score = s_chart[:, 1:, :, :self.NT].reshape(b, (n - 1) * n, self.NT, 1)
        # (b, n(n-1), ~~NT~~, hid)
        a_factor = ((a_hid * a_score.exp()) * _un(roots.exp(), [1, 3])).sum(dim=2)
        mem = mem_bc + a_factor
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
