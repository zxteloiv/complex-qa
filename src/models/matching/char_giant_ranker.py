import torch
from torch import nn
from torch.nn import functional as F
from utils.nn import seq_cross_ent, seq_likelihood

from .re2 import ChRE2
from .seq2seq_modeling import Seq2SeqModeling
from .seq_modeling import SeqModeling

class CharGiantRanker(nn.Module):
    def __init__(self,
                 a_embedding,
                 b_embedding,
                 re2: ChRE2,
                 seq_a2seq_b: Seq2SeqModeling,
                 seq_b2seq_a: Seq2SeqModeling,
                 seq_a: SeqModeling,
                 seq_b: SeqModeling,
                 a_padding: int,
                 b_padding: int,
                 a_ch_padding: int,
                 b_ch_padding: int,
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
        self.a_ch_pad = a_ch_padding
        self.b_pad = b_padding
        self.b_ch_pad = b_ch_padding
        self.dim_training_weight = .5

        self.loss_weighting = nn.Parameter(torch.zeros(5).float(), requires_grad=True)

    def forward(self,
                word_a: torch.LongTensor,
                word_b: torch.LongTensor,
                char_a: torch.LongTensor,
                char_b: torch.LongTensor,
                label: torch.LongTensor):
        """
        :param word_a: (N, a_len)
        :param word_b: (N, b_len)
        :param char_a: (N, a_len, a_char)
        :param char_b: (N, b_len, b_char)
        :param label: (N,)
        :return:
        """
        self.re2: ChRE2
        self.a2b: Seq2SeqModeling
        self.b2a: Seq2SeqModeling
        self.a_seq: SeqModeling
        self.b_seq: SeqModeling

        # word/char_a/b[_mask]: all the same shape
        input_masks = list(map(lambda x: (x != 0).long(), (word_a, word_b, char_a, char_b)))
        word_a_mask, word_b_mask, char_a_mask, char_b_mask = input_masks

        a_inp, b_inp, a_inp_mask, b_inp_mask = list(map(lambda t: t[:, :-1].contiguous(),
                                                        (word_a, word_b, word_a_mask, word_b_mask)))
        a_tgt, b_tgt, a_tgt_mask, b_tgt_mask = list(map(lambda t: t[:, 1:].contiguous(),
                                                        (word_a, word_b, word_a_mask, word_b_mask)))

        # ---- matching ------
        # matching logits: (batch, 2)
        matching_logits = self.re2(word_a, char_a, word_b, char_b)
        loss_m = F.cross_entropy(matching_logits, label)

        # ---- generation ----
        # a2b
        a_emb, b_emb = self.a_embedding(word_a), self.b_embedding(word_b)
        b_inp_emb = self.b_embedding(b_inp)
        logits_a2b = self.a2b.forward_emb(a_emb, b_inp_emb, word_a_mask, b_inp_mask)
        loss_a2b = F.mse_loss(seq_likelihood(logits_a2b, b_tgt, b_tgt_mask), label.float())

        # b2a
        a_inp_emb = self.a_embedding(a_inp)
        logits_b2a = self.b2a.forward_emb(b_emb, a_inp_emb, word_b_mask, a_inp_mask)
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
        pred_a = torch.cat([word_a[:, 0].unsqueeze(1), best_pred_a], dim=1)
        pred_b = torch.cat([word_b[:, 0].unsqueeze(1), best_pred_b], dim=1)
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

        return loss_normal + self.dim_training_weight * loss_dim.mean()

    def inference(self,
                  word_a: torch.LongTensor,
                  word_b: torch.LongTensor,
                  char_a: torch.LongTensor,
                  char_b: torch.LongTensor,
                  ):
        """
        :param word_a: (N, a_len)
        :param word_b: (N, b_len)
        :param char_a: (N, a_len, a_char)
        :param char_b: (N, b_len, b_char)
        :return:
        """
        self.re2: ChRE2
        self.a2b: Seq2SeqModeling
        self.b2a: Seq2SeqModeling
        self.a_seq: SeqModeling
        self.b_seq: SeqModeling

        # word/char_a/b[_mask]: all the same shape
        input_masks = list(map(lambda x: (x != 0).long(), (word_a, word_b, char_a, char_b)))
        word_a_mask, word_b_mask, char_a_mask, char_b_mask = input_masks

        a_inp, b_inp, a_inp_mask, b_inp_mask = list(map(lambda t: t[:, :-1].contiguous(),
                                                        (word_a, word_b, word_a_mask, word_b_mask)))
        a_tgt, b_tgt, a_tgt_mask, b_tgt_mask = list(map(lambda t: t[:, 1:].contiguous(),
                                                        (word_a, word_b, word_a_mask, word_b_mask)))

        # ---- matching ------
        # matching logits: (batch, 2)
        matching_logits = self.re2(word_a, char_a, word_b, char_b)
        ranking_m = torch.softmax(matching_logits, dim=-1)[:, 1]

        # ---- language model ----

        a_inp_emb = self.a_embedding(a_inp)
        # logits_a = self.a_seq.forward_emb(a_inp_emb, a_inp_mask)
        # ranking_a = seq_likelihood(logits_a, a_tgt, a_tgt_mask)

        b_inp_emb = self.b_embedding(b_inp)
        # logits_b = self.b_seq.forward_emb(b_inp_emb, b_inp_mask)
        # ranking_b = seq_likelihood(logits_b, b_tgt, b_tgt_mask)

        # ---- generation ----
        a_emb = self.a_embedding(word_a)
        b_emb = self.b_embedding(word_b)

        # a2b
        logits_a2b = self.a2b.forward_emb(a_emb, b_inp_emb, word_a_mask, b_inp_mask)
        ranking_a2b = seq_likelihood(logits_a2b, b_tgt, b_tgt_mask)

        # b2a
        logits_b2a = self.b2a.forward_emb(b_emb, a_inp_emb, word_b_mask, a_inp_mask)
        ranking_b2a = seq_likelihood(logits_b2a, a_tgt, a_tgt_mask)

        # return ranking_m, ranking_a, ranking_b, ranking_a2b, ranking_b2a
        return ranking_m, ranking_a2b, ranking_b2a

    def forward_loss_weight(self, *args):
        x = torch.stack(args, dim=1)
        prob = torch.softmax(self.loss_weighting[:3], dim=0)
        score = torch.matmul(x, prob)
        return score.log()


