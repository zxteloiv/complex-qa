from typing import Optional, List, Union, Tuple, Literal

import torch
import torch.nn as nn

from .C_PCFG import CompoundPCFG, unit_norm, _un
from .res import ResLayer
from ..interfaces.encoder import EmbedAndEncode


class TNPCFG(CompoundPCFG):
    def __init__(self,
                 rank: int,
                 num_nonterminal: int,
                 num_preterminal: int,
                 num_vocab_token: int,
                 hidden_sz: int,
                 emb_enc: EmbedAndEncode = None,
                 z_dim: Optional[int] = None,
                 encoding_dim: int = None,
                 padding_id: int = 0,
                 ):
        super(TNPCFG, self).__init__(num_nonterminal, num_preterminal, num_vocab_token,
                                     hidden_sz, emb_enc, z_dim, encoding_dim, padding_id,
                                     )
        self.r = rank

        self.rule_mlp = None    # decomposed and set None

        self.parent_mlp, self.left_mlp, self.right_mlp = (
            nn.Sequential(nn.Linear(self.hidden_sz + self.z_dim, self.hidden_sz),
                          ResLayer(self.hidden_sz, self.hidden_sz),
                          ResLayer(self.hidden_sz, self.hidden_sz),
                          nn.Linear(self.hidden_sz, self.r))
            for _ in range(3)
        )

    def get_pcfg_params(self, z):
        b = z.size()[0]

        root_emb = self.root_emb.expand(b, self.hidden_sz)
        root_emb = torch.cat([root_emb, z], -1)
        roots = self.root_mlp(root_emb).log_softmax(-1)

        z_term_ex = z.unsqueeze(1).expand(b, self.T, self.z_dim)
        z_nonterm_ex = z.unsqueeze(1).expand(b, self.NT, self.z_dim)

        term_emb = self.term_emb.unsqueeze(0).expand(b, self.T, self.hidden_sz)
        term_cat = torch.cat([term_emb, z_term_ex], -1)
        nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(b, self.NT, self.hidden_sz)
        nonterm_cat = torch.cat([nonterm_emb, z_nonterm_ex], dim=-1)

        # Pi_{P->w} (batch, T, V)
        term_logp = self.term_mlp(term_cat).log_softmax(-1)

        # head: (b, NT, r)
        head = self.parent_mlp(nonterm_cat).log_softmax(-1)

        # left/right: (b, NT_T, r)
        rule_cat = torch.cat([nonterm_cat, term_cat], dim=1)
        left = self.left_mlp(rule_cat).log_softmax(-2)
        right = self.right_mlp(rule_cat).log_softmax(-2)

        return roots, term_logp, (head, left, right)

    def inside(self, x, pcfg_params, x_hid=None):
        # x, x_mask: (batch, n)
        # x_hid: (batch, n, hid), must be of the same length as x
        # roots: (batch, NT)
        # terms: (batch, T, V)
        # head: (batch, NT, r), normalized on dim=-1
        # left/right: (batch, NT_T, r), normalized on dim=-2
        roots, term_logp, (head, left, right) = pcfg_params
        b, n = x.size()
        NTs = slice(0, self.NT)                         # NTs proceed all Ts

        # score: (b, n, n, NT_T)
        score, emb_chart = self._chart_init(x, pcfg_params, x_hid)
        # cat_hids: [(NT, hid), (T, hid), (NT_T, hid)]
        cat_hids = self._get_category_embeddings()

        device = x.device
        for width in range(1, n):
            coordinates = self._get_inside_coordinates(n, width, device)
            # b/c_score: (batch, pos=(n - width), arrangement=width, NT_T)
            b_score, c_score = self._inside_chart_select(score, coordinates)

            # left/right: (batch, 1, 1, NT_T, r) <- (batch, NT_T, r)
            # vb/wc_score: (batch, pos, arr, r)
            vb_score = (_un(left, [1, 2]) + _un(b_score, -1)).logsumexp(dim=-2)
            wc_score = (_un(right, [1, 2]) + _un(c_score, -1)).logsumexp(dim=-2)

            vbwc_score = vb_score + wc_score    # (b, pos, arr, r)
            uvbwc_score = _un(head, 1) + vbwc_score.logsumexp(dim=2, keepdim=True)  # (b, pos, NT, r)
            a_score = uvbwc_score.logsumexp(-1)                                     # (b, pos, NT)
            score[:, width, :n - width, NTs] = a_score

            if x_hid is not None:
                reduced_abc = self._write_emb_chart(emb_chart, coordinates, pcfg_params,
                                                    vbwc_score=vbwc_score, a_score=a_score, a_hid=a_hid)
                emb_chart[:, width, :n - width] = reduced_abc

        lengths = (x != self.padding).sum(-1)
        logPxs = score[torch.arange(b, device=device), lengths - 1, 0, NTs] + roots       # (b, NT)
        logPx = torch.logsumexp(logPxs, dim=1)  # sum out the start nonterm S
        if x_hid is not None:
            chart = emb_chart[:, 1:, :]     # (b, n - 1, n, NT)
            mem = chart.reshape(b, (n - 1) * n, self.encoding_dim)
            return logPx, mem
        return logPx

    def _write_emb_chart(self, emb_chart, coordinates, pcfg_params, **ctx):
        # to save storage, the computation is factored into 3 parts
        # recall score_arr: (batch, pos, A=NT, arrangement, B, C)
        roots, term, (head, left, right) = pcfg_params
        # vbwc_score: (b, pos, arr, r)
        # a_score: (b, pos, NT)
        # a_hid: (1, 1, NT, hid)
        vbwc_score, a_score, a_hid = list(map(ctx.get, ('vbwc_score', 'a_score', 'a_hid')))

        # *_hid: (batch, pos, arrangement, hid)
        b_repr, c_repr = self._inside_chart_select(emb_chart, coordinates)
        b_hid = self.binary_composer_left(b_repr)
        c_hid = self.binary_composer_right(c_repr)

        # vbwc_t: (b, pos, r, arr)
        vbwc_t = vbwc_score.transpose(-1, -2)
        # bc_hid: (b, pos, arr, hid)
        bc_hid = b_hid + c_hid

        # bc_hid: (b, pos, 1, r, hid)
        bc_hid_rs = torch.matmul(vbwc_t.exp(), bc_hid).unsqueeze(2)
        # head: (batch, NT, r), normalized on dim=-1
        # bc_factor: (b, pos, NT, ~~r~~, hid)
        bc_factor = (_un(head, [1, 4]).exp() * bc_hid_rs).sum(-2)

        # a_hid: (1, 1, A, hid)
        # a_score: (batch, pos, A)
        # a_factor: (b, pos, A, hid)
        a_factor = a_hid * _un(a_score, -1).exp()

        reduced_abc = self._nonterm_emb_reduction(a_factor + bc_factor, a_score, roots)
        normalized_abc = self.binary_norm(reduced_abc)
        return normalized_abc

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
