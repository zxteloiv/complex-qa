from typing import Optional, Tuple, Literal, Union, List

import torch
import torch.nn as nn

from .base_pcfg import PCFGMixin, PCFGModule
from .res import ResLayer
from ..modules.batched_stack import TensorBatchStack

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
                 encoding_out_dim: int = None,
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
        self.encoding_out_dim = encoding_out_dim or hidden_sz

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
            # compose embeddings according to unary and binary rules
            _hid_dim = encoding_out_dim
            self.unary_composer_word = nn.Sequential(nn.Linear(encoder_input_dim, _hid_dim),
                                                     ResLayer(_hid_dim, _hid_dim))
            self.binary_left_nonterm = ResLayer(_hid_dim, _hid_dim)
            self.binary_left_chart = ResLayer(_hid_dim, _hid_dim)
            self.binary_right_nonterm = ResLayer(_hid_dim, _hid_dim)
            self.binary_right_chart = ResLayer(_hid_dim, _hid_dim)
            self.binary_head = ResLayer(_hid_dim, _hid_dim)

            self.category_mapping = ResLayer(_hid_dim, _hid_dim)

        self._initialize()

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

        score = self._score_chart_init(x, pcfg_params)              # (b, n, n, NT_T)
        emb_chart = self._emb_chart_init(x, pcfg_params, x_hid)     # (b, n, n, ...)
        # cat_hids: [(NT, hid), (T, hid), (NT_T, hid)]
        cat_hids = self._get_category_embeddings()

        for width in range(1, n):
            coordinates = self.get_inside_coordinates(n, width, device)
            # b/c_score: (batch, pos=(n - width), arrangement=width, NT_T)
            b_score, c_score = self.inside_chart_select(score, coordinates)

            # score_arr: (batch, pos=(n-width), A=NT, arrangement=width, B=NT_T, C=NT_T)
            score_arr = _un(_un(b_score, -1) + _un(c_score, -2), 2) + _un(rule_logp, [1, 3])
            # a_score: (batch, pos, NT)
            a_score = torch.logsumexp(score_arr, dim=(3, 4, 5))
            score[:, width, :n - width, NTs] = a_score

            chart_layer = self._get_emb_chart_layer(emb_chart, coordinates, pcfg_params,
                                                    score_arr=score_arr, a_score=a_score, cat_hids=cat_hids,
                                                    b_score=b_score, c_score=c_score,
                                                    )
            emb_chart[:, width, :n - width] = unit_norm(chart_layer)

        lengths = (x != self.padding).sum(-1)
        logPxs = score[tr_(b), lengths - 1, 0, NTs] + roots       # (b, NT)
        logPx = torch.logsumexp(logPxs, dim=1)  # sum out the start NT
        if x_hid is not None:
            # fixing the length layer embedding
            chart = emb_chart[:, 1:, :]     # (b, n - 1, n, hid)
            mem = chart.reshape(b, (n - 1) * n, 1, self.encoding_out_dim)
            correction = _un(cat_hids[-1], [0, 1])
            corrected_mem = ((mem + correction) * _un(roots, [1, 3]).exp()).sum(dim=2)  # (b, n(n-1), hid)
            return logPx, corrected_mem
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

    def _emb_chart_init(self, x, pcfg_params, x_hid):
        if x_hid is None:
            return None

        b, n = x.size()
        emb_chart = x_hid.new_zeros(b, n, n, self.encoding_out_dim)
        emb_chart[:, 0] = self.unary_composer_word(x_hid)          # (b, n, hid)
        return emb_chart

    def _get_category_embeddings(self):
        a_hid = self.binary_head(self.nonterm_emb)          # (NT, hid)
        a_corr = self.category_mapping(self.nonterm_emb)    # (NT, hid)

        cat_emb = torch.cat([self.term_emb, self.nonterm_emb], dim=0)
        cat_corr = self.category_mapping(cat_emb)           # (NT_T, hid)

        b_hid = self.binary_left_nonterm(cat_corr)           # (NT_T, hid)
        c_hid = self.binary_right_nonterm(cat_corr)          # (NT_T, hid)
        return a_hid, b_hid, c_hid, a_corr

    def _get_emb_chart_layer(self, emb_chart, coordinates, pcfg_params, **ctx):
        #                      0     1    2         3       4  5
        # recall score_arr: (batch, pos, A=NT, arrangement, B, C)
        score_arr = ctx['score_arr']
        # a_hid/corr: (A=NT, hid), b/c_hid: (B/C=NT_T, hid)
        a_hid, b_hid, c_hid, a_corr = ctx['cat_hids']
        # a_score: (b, pos, A), b/c_score: (b, pos, arr, B/C)
        a_score, b_score, c_score = ctx['a_score'], ctx['b_score'], ctx['c_score']
        roots, _, _ = pcfg_params

        # prepare weights for different factors
        b_weight = score_arr.logsumexp(dim=5)       # (b, pos, A, arr, B)
        c_weight = score_arr.logsumexp(dim=4)       # (b, pos, A, arr, C)
        cell_weight = b_weight.logsumexp(dim=4)     # (b, pos, A, arr)

        cell_factor = self._get_cell_factor(emb_chart, coordinates, cell_weight)    # (b, pos, A, hid)
        b_factor = self._get_b_or_c_factor(b_hid, b_score, b_weight)
        c_factor = self._get_b_or_c_factor(c_hid, c_score, c_weight)
        a_factor = _un(a_hid, [0, 1]) * _un(a_score, -1).exp()

        # (b, pos, A, hid)
        layer_by_a = a_factor + b_factor + c_factor + cell_factor

        # add correction
        layer = (layer_by_a - _un(a_corr, [0, 1])).mean(dim=2)
        return layer    # (b, pos, hid)

    def _get_cell_factor(self, emb_chart, coordinates, bc_weight):
        # 1. Cell factor. extend size to (b, pos, A, arr, hid), then multiply and sum
        # l/r_cell, cell_hid: (b, pos, arr, hid)
        l_cell, r_cell = self.inside_chart_select(emb_chart, coordinates)
        cell_hid = self.binary_left_chart(l_cell) + self.binary_right_chart(r_cell)
        cell_hid_ex = _un(cell_hid, 2)              # (b, pos, 1, arr, hid)

        # bc_weight: (b, pos, A, arr)
        bc_weight_ex = _un(bc_weight, -1).exp()     # (b, pos, A, arr, 1)
        cell_factor = (cell_hid_ex * bc_weight_ex).sum(dim=-2)  # (b, pos, A, hid)
        return cell_factor  # (b, pos, A, hid)

    def _get_b_or_c_factor(self, cat_hid, cell_score, weight):
        # 2. B/C factor. extend size to (0b, 1pos, 2A, 3arr, 4B/C, 5hd), then multiply and sum
        # cat_hid: (B/C=NT_T, hid)
        # cell_score: (b, pos, arr, B/C)
        # weight: (b, pos, A, arr, B/C)
        cat_hid_ex = _un(cat_hid, [0, 1, 2, 3])
        score_ex = _un(cell_score, [2, 5]).exp()
        weight_ex = _un(weight, -1).exp()

        factor = (cat_hid_ex * score_ex * weight_ex).sum(dim=(3, 4))
        return factor   # (b, pos, A, hid)

    def _nonterm_emb_reduction(self, normalized_abc, a_score, roots):
        if self.nonterminal_reduction == 'mean':
            reduced_abc = normalized_abc.mean(dim=-2)
        elif self.nonterminal_reduction == 'norm_score':
            reduced_abc = (a_score.softmax(-1).unsqueeze(-1) * normalized_abc).sum(-2)
        elif self.nonterminal_reduction == 'root_score':
            # roots: (batch, NT) -> (batch, 1, NT, 1)
            reduced_abc = (roots.unsqueeze(1).exp().unsqueeze(-1) * normalized_abc).sum(-2)
        elif self.nonterminal_reduction == 'sum':
            reduced_abc = normalized_abc.sum(dim=-2)
        else:
            raise ValueError(f'unrecognized nonterminal reduction: {self.nonterminal_reduction}')
        return reduced_abc

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

    def get_encoder_output_size(self):
        return self.encoding_out_dim
