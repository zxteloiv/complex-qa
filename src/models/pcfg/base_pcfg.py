import torch
from .fn import stripe, diagonal_copy_, diagonal, checkpoint


class PCFGModule(torch.nn.Module):
    """
    A PCFG Module is the interface defined for the general-purpose PCFGs.
    Thus any inherited class had better not implement other interfaces defined for other model.

    e.g. an EmbedAndEncode implementation is restricted to be the encoder bundle of the Seq2seq,
    but a PCFG can be more than an encoder.
    therefore a PCFG extends (PCFGModule, EmbedAndEncode) are better avoided.
    """

    def inference(self, x):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def set_condition(self, conditions):
        raise NotImplementedError

    def generate(self, pcfg_params):
        raise NotImplementedError


class PCFGMixin:

    @torch.enable_grad()
    def decode(self, rules, lens, viterbi=False, mbr=False):
        return self._inside(rules=rules, lens=lens, viterbi=viterbi, mbr=mbr)

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        terms = rules['unary']
        rule = rules['rule']
        root = rules['root']

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            b_n_yz = contract(Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2), dim=2).reshape(batch, n, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1:, None, :]

            if w == 2:
                diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) +
                               span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            diagonal_copy_(s, contract(x, dim=0) +
                           span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr)
            return {'partition': logZ,
                    'prediction': prediction}

        else:
            return {'partition': logZ}

    def _get_prediction(self, logZ, span_indicator, lens, mbr=False):
        batch, seq_len = span_indicator.shape[:2]
        prediction = [[] for _ in range(batch)]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            assert logZ.requires_grad
            logZ.sum().backward()
            marginals = span_indicator.grad
            if mbr:
                return self._cky_zero_order(marginals.detach(), lens)
            else:
                viterbi_spans = marginals.nonzero().tolist()
                for span in viterbi_spans:
                    prediction[span[0]].append((span[1], span[2]))
        return prediction

    @torch.no_grad()
    def _cky_zero_order(self, marginals, lens):
        N = marginals.shape[-1]
        s = marginals.new_zeros(*marginals.shape).fill_(-1e9)
        p = marginals.new_zeros(*marginals.shape).long()
        diagonal_copy_(s, diagonal(marginals, 1), 1)
        for w in range(2, N):
            n = N - w
            starts = p.new_tensor(range(n))
            if w != 2:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            else:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            X, split = (Y + Z).max(2)
            x = X + diagonal(marginals, w)
            diagonal_copy_(s, x, w)
            diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)

        def backtrack(p, i, j):
            if j == i + 1:
                return [(i, j)]
            split = p[i][j]
            ltree = backtrack(p, i, split)
            rtree = backtrack(p, split, j)
            return [(i, j)] + ltree + rtree

        p = p.tolist()
        lens = lens.tolist()
        spans = [backtrack(p[i], 0, length) for i, length in enumerate(lens)]
        return spans
