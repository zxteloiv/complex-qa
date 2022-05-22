from typing import Optional

import torch
import torch.nn as nn
from allennlp.nn.util import masked_max

from .base_pcfg import PCFGMixin, PCFGModule
from .res import ResLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from ..interfaces.encoder import EmbedAndEncode


def unit_norm(x, p=2, eps=1e-12):
    return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


class CompoundPCFG(PCFGModule):
    def __init__(self,
                 num_nonterminal: int,
                 num_preterminal: int,
                 num_vocab_token: int,
                 hidden_sz: int,
                 emb_enc: EmbedAndEncode,
                 z_dim: Optional[int] = None,
                 encoding_dim: int = None,
                 padding_id: int = 0,
                 ):
        super().__init__()

        self.NT = num_nonterminal
        self.T = num_preterminal
        self.V = num_vocab_token
        self.padding = padding_id

        self.hidden_sz = hidden_sz
        self.z_dim = z_dim or hidden_sz
        self.encoding_dim = encoding_dim or hidden_sz

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

        # re-use the RNN and embeddings of the inference net, to build a new
        self.emb_enc = emb_enc
        self.infer_net = nn.Linear(emb_enc.get_output_dim(), self.z_dim * 2)

        self.NT_T = self.NT + self.T
        self.rule_mlp = nn.Linear(compound_sz, self.NT_T ** 2)

        # compose embeddings according to unary and binary rules
        self.unary_composer_word = nn.Sequential(nn.Linear(emb_enc.get_output_dim(), encoding_dim),
                                                 ResLayer(encoding_dim, encoding_dim))
        self.unary_composer_term = ResLayer(encoding_dim, encoding_dim)
        self.binary_composer_child = ResLayer(encoding_dim, encoding_dim)
        self.binary_composer_head = ResLayer(encoding_dim, encoding_dim)
        self.activation = nn.Mish()

        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def encode(self, x: torch.Tensor):
        assert x is not None
        # q_*: (batch, z_dim)
        x_hid, q_mean, q_logvar = self.run_inference(x)

        kl = self.kl(q_mean, q_logvar).sum(1)
        z = self.reparameterized_sample(q_mean, q_logvar)
        pcfg_params = self.get_pcfg_params(z)

        logPx, x_tree_hid = self.inside(x, pcfg_params, x_hid)
        return logPx, kl, x_tree_hid

    def run_inference(self, x):
        layered_state, state_mask = self.emb_enc(x)     # (b, N, d)
        state = layered_state[-1]
        pooled = masked_max(state, state_mask.bool().unsqueeze(-1), dim=1)     # (b, d)
        q_mean, q_logvar = self.infer_net(pooled).split(self.z_dim, dim=1)
        return state, q_mean, q_logvar

    def reparameterized_sample(self, q_mean, q_logvar):
        z = q_mean  # use q_mean to approximate during evaluation
        if self.training:
            noise = q_mean.new_zeros(q_mean.size()).normal_(0, 1)
            z = q_mean + (.5 * q_logvar).exp() * noise  # z = u + sigma * noise
        return z

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
        x_mask = (x != self.padding).long()
        score = x.new_full((b, n, n, self.NT_T), -1e9)  # the chart jointly saves nonterminals and preterminals
        NTs = slice(0, self.NT)                         # NTs proceed all Ts
        Ts = slice(self.NT, self.NT_T)
        un_ = torch.unsqueeze    # alias, for brevity
        tr_ = torch.arange

        # 0, (batch, n, T)
        level0 = un_(term_logp, 1).expand(b, n, self.T, self.V).gather(
            index=un_(un_(x, -1).expand(b, n, self.T), -1), dim=-1,
        ).squeeze(-1)
        score[:, 0, :, Ts] = level0

        emb_chart = None
        if x_hid is not None:
            emb_chart = x_hid.new_zeros(b, n, n, self.NT_T, self.hidden_sz)
            term_hid = self.unary_composer_term(self.term_emb)  # (T, hid)
            word_hid = self.unary_composer_word(x_hid)          # (b, n, hid)
            emb_chart[:, 0, :, Ts] = unit_norm(un_(un_(term_hid, 0), 1) + un_(word_hid, 2))   # (b, n, T, hid)

        device = x.device
        for width in range(1, n):
            print(f'inside: width={width}')
            lvl_b = un_(tr_(width, device=device), 0)                 # (pos=1, sublvl)
            pos_b = un_(tr_(n - width, device=device), 1)             # (pos, subpos=1)
            lvl_c = un_(tr_(width - 1, -1, -1, device=device), 0)     # (pos=1, sublvl), reversed lvl_b
            pos_c = un_(tr_(1, width + 1, device=device), 0) + pos_b  # (pos=(n-width), subpos=width))

            # *_score: (batch, pos=(n - width), arrangement=width, NT_T, NT_T)
            b_score = score[:, lvl_b, pos_b, :, None].clone()
            c_score = score[:, lvl_c, pos_c, None, :].clone()
            # score_arr: (batch, pos=(n-width), A=NT, arrangement=width, B=NT_T, C=NT_T)
            score_arr = un_(b_score + c_score, 2) + un_(un_(rule_logp, 1), 3)
            # a_score: (batch, pos, NT)
            a_score = torch.logsumexp(score_arr, dim=(3, 4, 5))
            score[:, width, :n - width, NTs] = a_score

            def inspect_mem(i=0):
                mem = torch.cuda.memory_reserved(0)
                if i > 0:
                    print(f'width={width} i={i} memory={mem}')
                else:
                    print(f'width={width} memory={mem}')

            inspect_mem(0)

            if x_hid is not None:
                b_hid = emb_chart[:, lvl_b, pos_b, :, None].clone()
                c_hid = emb_chart[:, lvl_c, pos_c, None, :].clone()
                # bc_hid: (batch, pos, arrangement, B, C, hid)
                bc_hid = self.binary_composer_child(b_hid) + self.binary_composer_child(c_hid)
                # a_hid: (1, 1, NT, 1, 1, 1, hid)
                a_hid = self.binary_composer_head(self.nonterm_emb)[None, None, :, None, None, None]     # (NT, hid)
                # abc_weight: (batch, pos, A, arrangement, B, C, 1)
                abc_weight = un_(score_arr.exp(), -1)
                inspect_mem(1)
                # normalized_abc = (abc_weight * unit_norm(a_hid + un_(bc_hid, 2))).sum(dim=(3, 4, 5))
                tmp = a_hid + un_(bc_hid, 2)
                inspect_mem(2)
                tmp = unit_norm(tmp)
                inspect_mem(3)
                tmp = abc_weight * tmp
                inspect_mem(4)
                normalized_abc = tmp.sum(dim=(3, 4, 5))
                inspect_mem(5)
                emb_chart[:, width, :n - width, NTs] = normalized_abc

        lengths = x_mask.sum(-1)
        logPxs = score[tr_(b, device=device), lengths - 1, 0] + roots       # (b, NT)
        logPx = torch.logsumexp(logPxs, dim=1)  # sum out the start NT
        if x_hid is not None:
            root_hid = emb_chart[tr_(b, device=device), lengths - 1, 0]     # (b, NT, hid)
            x_tree_hid = (un_(logPxs, -1).exp() * root_hid).sum(dim=1)      # (b, hid)
            return logPx, x_tree_hid
        else:
            return logPx

    def set_condition(self, conditions):
        pass

    def conditioned_generate(self, conditions=None):
        pass

    @staticmethod
    def kl(mean, logvar):
        # mean, logvar: (batch, z_dim)
        result = -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)
        return result

    def forward(self, input, evaluating=False):
        x = input['word']
        b, n = x.shape[:2]
        seq_len = input['seq_len']

        def enc(x):
            x_embbed = self.embedding(x)
            x_packed = pack_padded_sequence(
                x_embbed, seq_len, batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.enc_rnn(x_packed)
            padding_value = float("-inf")
            output, lengths = pad_packed_sequence(
                h_packed, batch_first=True, padding_value=padding_value
            )
            h = output.max(1)[0]
            out = self.enc_out(h)
            mean = out[:, : self.z_dim]
            lvar = out[:, self.z_dim :]
            return mean, lvar

        mean, lvar = enc(x)
        z = mean

        if not evaluating:
            z = mean.new(b, mean.size(1)).normal_(0,1)
            z = (0.5 * lvar).exp() * z + mean


        def roots():
            root_emb = self.root_emb.expand(b, self.hidden_sz)
            root_emb = torch.cat([root_emb, z], -1)
            roots = self.root_mlp(root_emb).log_softmax(-1)
            return roots

        def terms():
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                b, n, self.T, self.hidden_sz
            )
            z_expand = z.unsqueeze(1).expand(b, n, self.z_dim)
            z_expand = z_expand.unsqueeze(2).expand(b, n, self.T, self.z_dim)
            term_emb = torch.cat([term_emb, z_expand], -1)
            term_prob = self.term_mlp(term_emb).log_softmax(-1)
            indices = x.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules():
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                b, self.NT, self.hidden_sz
            )
            z_expand = z.unsqueeze(1).expand(
                b, self.NT, self.z_dim
            )
            nonterm_emb = torch.cat([nonterm_emb, z_expand], -1)
            rule_prob = self.rule_mlp(nonterm_emb).log_softmax(-1)
            rule_prob = rule_prob.reshape(b, self.NT, self.NT_T, self.NT_T)
            return rule_prob


        root, unary, rule = roots(), terms(), rules()

        return {'unary': unary,
                'root': root,
                'rule': rule,
                'kl': self.kl(mean, lvar).sum(1)}

    def loss(self, input):
        rules = self.forward(input)
        result = self.pcfg._inside(rules=rules, lens=input['seq_len'])
        loss = (-result['partition'] + rules['kl']).mean()
        return loss

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

        result['partition'] -= rules['kl']
        return result


