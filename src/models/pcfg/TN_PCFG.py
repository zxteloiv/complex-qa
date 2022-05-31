from typing import Optional, List, Union

import torch
import torch.nn as nn

from .C_PCFG import CompoundPCFG, unit_norm
from .res import ResLayer
from ..interfaces.encoder import EmbedAndEncode


class TNPCFG(CompoundPCFG):
    def __init__(self,
                 rank: int,
                 num_nonterminal: int,
                 num_preterminal: int,
                 num_vocab_token: int,
                 hidden_sz: int,
                 emb_enc: EmbedAndEncode,
                 z_dim: Optional[int] = None,
                 encoding_dim: int = None,
                 padding_id: int = 0,
                 ):
        super(TNPCFG, self).__init__(num_nonterminal, num_preterminal, num_vocab_token,
                                     hidden_sz, emb_enc, z_dim, encoding_dim, padding_id)
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
        score = x.new_full((b, n, n, self.NT_T), -1e9)  # the chart jointly saves nonterminals and preterminals
        NTs = slice(0, self.NT)                         # NTs proceed all Ts
        Ts = slice(self.NT, self.NT_T)

        def un_(x: torch.Tensor, dims: Union[List[int], int]):
            if isinstance(dims, int):
                return x.unsqueeze(dims)
            else:
                for d in sorted(dims):
                    x = x.unsqueeze(d)
                return x

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
            # b/c_score: (batch, pos=(n - width), arrangement=width, NT_T)
            b_score, c_score = self._inside_chart_select(score, coordinates)

            # left/right: (batch, 1, 1, NT_T, r, 1) <- (batch, NT_T, r)
            # vb/wc_score: (batch, pos, arr, r)
            vb_score = (un_(left, [1, 2]) + un_(b_score, -1)).logsumexp(dim=-2)
            wc_score = (un_(right, [1, 2]) + un_(c_score, -1)).logsumexp(dim=-2)

            vbwc_score = (vb_score + wc_score).logsumexp(dim=-2).unsqueeze(2)   # (b, pos, 1, r)
            uvbwc_score = un_(head, 1) + vbwc_score                             # (b, pos, NT, r)
            a_score = uvbwc_score.logsumexp(-1)                                 # (b, pos, NT)
            score[:, width, :n - width, NTs] = a_score

            if x_hid is not None:
                # to save storage, the computation is factored into 3 parts
                # recall score_arr: (batch, pos, A=NT, arrangement, B, C)

                # b/c_hid: (batch, pos, arrangement, B/C, hid)
                b_repr, c_repr = self._inside_chart_select(emb_chart, coordinates)
                b_hid = un_(b_score, -1) * self.binary_composer_left(b_repr)
                c_hid = un_(c_score, -1) * self.binary_composer_right(c_repr)

                # left/right: (batch, 1, 1, NT_T, r, 1) <- (batch, NT_T, r)
                vb_hid = (un_(left, [1, 2, 5]).exp() * un_(b_hid, -2)).sum(3)   # (batch, pos, arr, ~~B/C~~, r, hid)
                wc_hid = (un_(right, [1, 2, 5]).exp() * un_(c_hid, -2)).sum(3)  # -> (batch, pos, arr,       r, hid)

                # (batch, pos, ~~arr~~, r, hid)
                vb_hid_wc = (vb_hid * un_(wc_score, -1).exp()).sum(2)
                vb_wc_hid = (un_(vb_score, -1).exp() * wc_hid).sum(2)

                # head: (batch, NT, r)
                # b/c_factor: (batch, pos, r, hid)
                bc_factor = torch.matmul(un_(head, 1), vb_hid_wc + vb_wc_hid)

                # a_hid: (1, 1, A, hid)
                # a_score: (batch, pos, A)
                a_factor = a_hid * un_(a_score, -1).exp()

                normalized_abc = unit_norm(a_factor + bc_factor)
                emb_chart[:, width, :n - width, NTs] = normalized_abc

        lengths = (x != self.padding).sum(-1)
        logPxs = score[torch.arange(b, device=device), lengths - 1, 0, NTs] + roots       # (b, NT)
        logPx = torch.logsumexp(logPxs, dim=1)  # sum out the start nonterm S
        if x_hid is not None:
            chart = emb_chart[:, 1:, :, NTs]     # (b, n - 1, n, NT, hid)
            mem = (roots.reshape(b, 1, 1, self.NT, 1) * chart).sum(3).reshape(b, (n - 1) * n, -1)
            return logPx, mem
        return logPx

    def forward(self, input, **kwargs):
        x = input['word']
        b, n = x.shape[:2]

        def roots():
            roots = self.root_mlp(self.root_emb).log_softmax(-1)
            return roots.expand(b, roots.shape[-1]).contiguous()

        def terms():
            term_prob = self.term_mlp(self.term_emb).log_softmax(-1)
            term_prob = term_prob.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.V
            )
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[:self.NT]
            head = self.parent_mlp(nonterm_emb).log_softmax(-1)
            left = self.left_mlp(rule_state_emb).log_softmax(-2)
            right = self.right_mlp(rule_state_emb).log_softmax(-2)
            head = head.unsqueeze(0).expand(b, *head.shape)
            left = left.unsqueeze(0).expand(b, *left.shape)
            right = right.unsqueeze(0).expand(b, *right.shape)
            return head, left, right

        root, unary, (head, left, right) = roots(), terms(), rules()

        return {'unary': unary,
                'root': root,
                'head': head,
                'left': left,
                'right': right,
                'kl': 0}

    def loss(self, input):
        rules = self.forward(input)
        result = self.pcfg._inside(rules=rules, lens=input['seq_len'])
        logZ = -result['partition'].mean()
        return logZ

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == 'viterbi':
            raise NotImplementedError("TD-PCFG cannot be downgraded back to viterbi decoding by design.")

        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError
