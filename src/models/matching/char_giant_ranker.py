from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from utils.nn import seq_cross_ent, seq_likelihood, prepare_input_mask

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

    def forward(self,
                word_a: torch.LongTensor,
                word_b: torch.LongTensor,
                char_a: torch.LongTensor,
                char_b: torch.LongTensor,
                label: Optional[torch.LongTensor] = None,
                rank: Optional[torch.Tensor] = None,
                return_repr: bool = False,
                ):
        """
        :param word_a: (N, a_len)
        :param word_b: (N, b_len)
        :param char_a: (N, a_len, a_char)
        :param char_b: (N, b_len, b_char)
        :param label: (N,)
        :param rank: (N, 1)
        :return:
        """
        # ---- matching ------
        self.re2: ChRE2

        inp_tensors = self._prepare_input(word_a, word_b, char_a, char_b)
        word_a_mask, word_b_mask, char_a_mask, char_b_mask = inp_tensors[:4]

        a = self.re2.a_emb(word_a, char_a, char_a_mask)
        b = self.re2.b_emb(word_b, char_b, char_b_mask)

        a, b = self.re2.forward_encoding(a, b, word_a_mask, word_b_mask)

        if rank is not None:
            rank = (rank + 1).float().reciprocal_()     # to transform ranks such that greater ranker is better

        _, likelihoods = self._forward_generation_model(word_a, word_b, inp_tensors)

        assert all(map(lambda l: l.dim() > 0, likelihoods))
        likelihoods_expand = [l.unsqueeze(1) if l.dim() == 1 else l for l in likelihoods]

        if label is None:   # evaluation
            # matching logits: (batch, 2)
            matching_logits, features = self.re2.prediction(a, b, rank, *likelihoods_expand, return_repr=True)
            reranking_score = torch.softmax(matching_logits, dim=-1)[:, 1]
            score_tuple = (reranking_score,) + tuple(likelihoods)

            if return_repr:
                return score_tuple, features
            else:
                return score_tuple

        else:   # training
            matching_logits, features = self.re2.prediction(a, b, rank, *likelihoods_expand, return_repr=True)
            loss_m = F.cross_entropy(matching_logits, label, reduction="none")
            loss_reg = self._model_reg_loss(word_a, word_b, inp_tensors, label)
            batch_loss = loss_m + loss_reg
            if return_repr:
                return batch_loss, features
            else:
                return batch_loss.mean()

    def _prepare_input(self, word_a, word_b, char_a, char_b):
        _, word_a_mask = prepare_input_mask(word_a, self.a_pad)
        _, char_a_mask = prepare_input_mask(char_a, self.a_ch_pad)
        _, word_b_mask = prepare_input_mask(word_b, self.b_pad)
        _, char_b_mask = prepare_input_mask(char_b, self.b_ch_pad)

        vecs = [word_a_mask, word_b_mask, char_a_mask, char_b_mask]
        vecs += list(map(lambda t: t[:, :-1].contiguous(), (word_a, word_b, word_a_mask, word_b_mask)))
        vecs += list(map(lambda t: t[:, 1:].contiguous(), (word_a, word_b, word_a_mask, word_b_mask)))
        return vecs

    def _forward_generation_model(self, word_a, word_b, inp_tensors, label: Optional[torch.Tensor] = None):
        """return logits of generation model, if label is not None, append the generation loss"""
        word_a_mask, word_b_mask, char_a_mask, char_b_mask = inp_tensors[:4]
        a_inp, b_inp, a_inp_mask, b_inp_mask = inp_tensors[4:8]
        a_tgt, b_tgt, a_tgt_mask, b_tgt_mask = inp_tensors[8:]

        # generation models are based on only word embedding
        # otherwise char-embeddings will be required for dual training.
        a_emb, b_emb = self.a_embedding(word_a), self.b_embedding(word_b)

        # a2b
        b_inp_emb = self.b_embedding(b_inp)
        logits_a2b = self.a2b.forward_emb(a_emb, b_inp_emb, word_a_mask, b_inp_mask)
        likelihood_a2b = seq_likelihood(logits_a2b, b_tgt, b_tgt_mask)
        # b2a
        a_inp_emb = self.a_embedding(a_inp)
        logits_b2a = self.b2a.forward_emb(b_emb, a_inp_emb, word_b_mask, a_inp_mask)
        likelihood_b2a = seq_likelihood(logits_b2a, a_tgt, a_tgt_mask)

        if label is None:
            return [logits_a2b, logits_b2a], [likelihood_a2b, likelihood_b2a]
        else:
            loss_a2b = F.mse_loss(likelihood_a2b, label.float(), reduction='none')
            loss_b2a = F.mse_loss(likelihood_b2a, label.float(), reduction='none')
            loss_gen = loss_b2a + loss_a2b
            return [logits_a2b, logits_b2a], [likelihood_a2b, likelihood_b2a], loss_gen

    def _model_reg_loss(self, word_a, word_b, inp_tensors, label):
        word_a_mask, word_b_mask, char_a_mask, char_b_mask = inp_tensors[:4]
        a_inp, b_inp, a_inp_mask, b_inp_mask = inp_tensors[4:8]
        a_tgt, b_tgt, a_tgt_mask, b_tgt_mask = inp_tensors[8:]

        # ------------------------
        # ---- generation ----
        # a2b
        logits, likelihoods, loss_gen = self._forward_generation_model(word_a, word_b, inp_tensors, label)
        a_inp_emb, b_inp_emb = list(map(self.a_embedding, (a_inp, b_inp)))
        logits_a2b, logits_b2a = logits

        # ------------------------
        # ---- language model ----
        logits_a = self.a_seq.forward_emb(a_inp_emb, a_inp_mask)
        loss_a = seq_cross_ent(logits_a, a_tgt, a_tgt_mask, average=None)

        logits_b = self.b_seq.forward_emb(b_inp_emb, b_inp_mask)
        loss_b = seq_cross_ent(logits_b, b_tgt, b_tgt_mask, average=None)

        loss_lm = loss_a + loss_b

        # ------------------------
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

        batch_loss = loss_lm + loss_gen + self.dim_training_weight * loss_dim
        return batch_loss

