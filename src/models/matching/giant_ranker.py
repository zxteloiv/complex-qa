import torch
from torch import nn
from torch.nn import functional as F
from utils.nn import seq_cross_ent, seq_likelihood

from .re2 import RE2
from .seq2seq_modeling import Seq2SeqModeling
from .seq_modeling import SeqModeling

class GiantRanker(nn.Module):
    def __init__(self,
                 a_embedding,
                 b_embedding,
                 re2: RE2,
                 seq_a2seq_b: Seq2SeqModeling,
                 seq_b2seq_a: Seq2SeqModeling,
                 seq_a: SeqModeling,
                 seq_b: SeqModeling,
                 a_padding: int,
                 b_padding: int,
                 ):
        super().__init__()
        self.re2 = re2
        self.a2b: Seq2SeqModeling = seq_a2seq_b
        self.b2a: Seq2SeqModeling = seq_b2seq_a
        self.a_seq: SeqModeling = seq_a
        self.b_seq: SeqModeling = seq_b
        self.a_embedding = a_embedding
        self.b_embedding = b_embedding
        self.a_pad = a_padding
        self.b_pad = b_padding
        self.dim_training_weight = .5

        self.loss_weighting = nn.Parameter(torch.zeros(5).float(), requires_grad=True)

    def forward(self, sent_a: torch.LongTensor, sent_b: torch.LongTensor, label: torch.LongTensor):
        """
        :param sent_a: (N, a_len)
        :param sent_b: (N, b_len)
        :param label: (N,)
        :return:
        """
        self.re2: RE2
        self.a2b: Seq2SeqModeling
        self.b2a: Seq2SeqModeling
        self.a_seq: SeqModeling
        self.b_seq: SeqModeling

        a_inp, b_inp = sent_a[:, :-1].contiguous(), sent_b[:, :-1].contiguous()
        a_tgt, b_tgt = sent_a[:, 1:].contiguous(), sent_b[:, 1:].contiguous()
        a_inp_mask = (a_inp != self.a_pad).long()
        b_inp_mask = (b_inp != self.b_pad).long()
        a_tgt_mask = (a_tgt != self.a_pad).long()
        b_tgt_mask = (b_tgt != self.b_pad).long()
        a_mask = (sent_a != self.a_pad).long()
        b_mask = (sent_b != self.b_pad).long()

        # ---- matching ------
        a_emb, b_emb = list(map(self.a_embedding, (sent_a, sent_b)))
        # matching logits: (batch, 2)
        matching_logits = self.re2.forward_embs(a_emb, b_emb, a_mask, b_mask)
        loss_m = F.cross_entropy(matching_logits, label)

        # ---- generation ----
        # a2b
        b_inp_emb = self.b_embedding(b_inp)
        logits_a2b = self.a2b.forward_emb(a_emb, b_inp_emb, a_mask, b_inp_mask)
        loss_a2b = F.mse_loss(seq_likelihood(logits_a2b, b_tgt, b_tgt_mask), label.float())

        # b2a
        a_inp_emb = self.a_embedding(a_inp)
        logits_b2a = self.b2a.forward_emb(b_emb, a_inp_emb, b_mask, a_inp_mask)
        loss_b2a = F.mse_loss(seq_likelihood(logits_b2a, a_tgt, a_tgt_mask), label.float())

        # ---- language model ----
        logits_a = self.a_seq.forward_emb(a_inp_emb, a_inp_mask)
        loss_a = seq_cross_ent(logits_a, a_tgt, a_tgt_mask, average=None)

        logits_b = self.b_seq.forward_emb(b_inp_emb, b_inp_mask)
        loss_b = seq_cross_ent(logits_b, b_tgt, b_tgt_mask, average=None)

        loss_normal = loss_m + loss_a.mean() + loss_b.mean() + loss_b2a.mean() + loss_a2b.mean()
        # return loss_normal

        # ---- dual model -----
        logprob_b2a = torch.log_softmax(logits_b2a, dim=-1)
        logprob_a2b = torch.log_softmax(logits_a2b, dim=-1)
        # best_pred: (N, len - 1)
        best_pred_a_logprob, best_pred_a = torch.max(logprob_b2a, dim=-1)
        best_pred_b_logprob, best_pred_b = torch.max(logprob_a2b, dim=-1)
        # pred_a: (N, a_len)
        # pred_b: (N, b_len)
        pred_a = torch.cat([sent_a[:, 0].unsqueeze(1), best_pred_a], dim=1)
        pred_b = torch.cat([sent_b[:, 0].unsqueeze(1), best_pred_b], dim=1)
        pred_a_mask = (pred_a != self.a_pad).long()
        pred_b_mask = (pred_b != self.b_pad).long()

        logits_pred_a2b = self.a2b.forward_emb(self.a_embedding(pred_a), b_inp_emb, pred_a_mask, b_inp_mask)
        reward_pred_a2b = seq_likelihood(logits_pred_a2b, b_tgt, b_tgt_mask)
        logits_pred_b2a = self.b2a.forward_emb(self.b_embedding(pred_b), a_inp_emb, pred_b_mask, a_inp_mask)
        reward_pred_b2a = seq_likelihood(logits_pred_b2a, a_tgt, a_tgt_mask)

        pred_a_inp, pred_b_inp = pred_a[:, :-1].contiguous(), pred_b[:, :-1].contiguous()
        pred_a_tgt, pred_b_tgt = pred_a[:, 1:].contiguous(), pred_b[:, 1:].contiguous()
        pred_a_inp_mask = (pred_a_inp != self.a_pad).long()
        pred_b_inp_mask = (pred_b_inp != self.b_pad).long()
        pred_a_tgt_mask = (pred_a_tgt != self.a_pad).long()
        pred_b_tgt_mask = (pred_b_tgt != self.b_pad).long()

        logits_pred_a = self.a_seq.forward_emb(self.a_embedding(pred_a_inp), pred_a_inp_mask)
        reward_pred_a = seq_likelihood(logits_pred_a, pred_a_tgt, pred_a_tgt_mask)
        logits_pred_b = self.b_seq.forward_emb(self.b_embedding(pred_b_inp), pred_b_inp_mask)
        reward_pred_b = seq_likelihood(logits_pred_b, pred_b_tgt, pred_b_tgt_mask)

        # baseline omitted
        # reward: (batch,)
        a_reward = reward_pred_a + reward_pred_a2b
        b_reward = reward_pred_b + reward_pred_b2a

        # reward transform
        a_reward = torch.sigmoid(a_reward / 2.)
        b_reward = torch.sigmoid(b_reward / 2.)

        loss_dim_a = - a_reward * best_pred_a_logprob.sum(dim=-1)
        loss_dim_b = - b_reward * best_pred_b_logprob.sum(dim=-1)

        loss_dim = loss_dim_a + loss_dim_b  # without using an EM-analogous opt.

        return loss_normal + 0.5 * loss_dim.mean()

    def inference(self, sent_a, sent_b):
        self.re2: RE2
        self.a2b: Seq2SeqModeling
        self.b2a: Seq2SeqModeling
        self.a_seq: SeqModeling
        self.b_seq: SeqModeling

        a_inp, b_inp = sent_a[:, :-1].contiguous(), sent_b[:, :-1].contiguous()
        a_tgt, b_tgt = sent_a[:, 1:].contiguous(), sent_b[:, 1:].contiguous()
        a_inp_mask = (a_inp != self.a_pad).long()
        b_inp_mask = (b_inp != self.b_pad).long()
        a_tgt_mask = (a_tgt != self.a_pad).long()
        b_tgt_mask = (b_tgt != self.b_pad).long()
        a_mask = (sent_a != self.a_pad).long()
        b_mask = (sent_b != self.b_pad).long()

        # ---- matching ------
        a_emb, b_emb = list(map(self.a_embedding, (sent_a, sent_b)))
        # matching logits: (batch, 2)
        matching_logits = self.re2.forward_embs(a_emb, b_emb, a_mask, b_mask)
        ranking_m = torch.softmax(matching_logits, dim=-1)[:, 1]

        # ---- language model ----

        a_inp_emb = self.a_embedding(a_inp)
        # logits_a = self.a_seq.forward_emb(a_inp_emb, a_inp_mask)
        # ranking_a = seq_likelihood(logits_a, a_tgt, a_tgt_mask)

        b_inp_emb = self.b_embedding(b_inp)
        # logits_b = self.b_seq.forward_emb(b_inp_emb, b_inp_mask)
        # ranking_b = seq_likelihood(logits_b, b_tgt, b_tgt_mask)

        # ---- generation ----
        # a2b
        logits_a2b = self.a2b.forward_emb(a_emb, b_inp_emb, a_mask, b_inp_mask)
        ranking_a2b = seq_likelihood(logits_a2b, b_tgt, b_tgt_mask)

        # b2a
        logits_b2a = self.b2a.forward_emb(b_emb, a_inp_emb, b_mask, a_inp_mask)
        ranking_b2a = seq_likelihood(logits_b2a, a_tgt, a_tgt_mask)

        # return ranking_m, ranking_a, ranking_b, ranking_a2b, ranking_b2a
        return ranking_m, ranking_a2b, ranking_b2a

    def forward_loss_weight(self, *args):
        x = torch.stack(args, dim=1)
        prob = torch.softmax(self.loss_weighting[:3], dim=0)
        score = torch.matmul(x, prob)
        return score.log()


