from typing import Optional, Tuple, Generic
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from utils.nn import seq_cross_ent, seq_likelihood
from models.modules.stacked_encoder import StackedEncoder
from .ebnf_npda import NeuralEBNF
from allennlp.training.metrics.perplexity import Perplexity, Average
from utils.nn import prepare_input_mask

Tuple3Tensor = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Tuple3LongTensor = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
Tuple3FloatTensor = Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
Tuple4Tensor = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
Tuple4LongTensor = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]
Tuple4FloatTensor = Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
Tuple5Tensor = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
Tuple5LongTensor = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]
Tuple5FloatTensor = Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]


class Seq2PDA(nn.Module):
    def __init__(self,
                 encoder: StackedEncoder,
                 src_embedding: nn.Embedding,
                 enc_attn_net: nn.Module,
                 ebnf_npda: NeuralEBNF,
                 tok_pad_id: int,
                 nt_fi: int,
                 inference_with_oracle_tree: bool = True
                 ):
        super().__init__()
        self.encoder = encoder
        self.src_embedder = src_embedding
        self.enc_attn_net = enc_attn_net

        self.ppl = Perplexity()
        self.err = Average()
        self.ebnf_npda = ebnf_npda
        self.tok_pad = tok_pad_id
        self.nt_fi = nt_fi

        self.inference_with_oracle_tree = inference_with_oracle_tree

    def get_metric(self, reset=False):
        ppl = self.ppl.get_metric(reset)
        err = self.err.get_metric(reset)
        return {"PPL": ppl, "ERR": err}

    get_metrics = get_metric

    def forward(self, source_tokens, derivation_tree: torch.LongTensor, token_fidelity: torch.LongTensor):
        """
        :param source_tokens: (batch, src_len)
        :param derivation_tree: (batch, derivation, seq), seq: [lhs, *rhs]
        :param token_fidelity: (batch, derivation, rhs_seq), rhs_seq = seq - 1, rhs: [<RuleGo> ... <RuleEnd>]
        :return:
        """
        if source_tokens is not None:
            source_tokens, source_mask = prepare_input_mask(source_tokens, self.tok_pad)
            source = self.src_embedder(source_tokens)
            source_hidden, _ = self.encoder(source, source_mask)
            enc_attn_fn = lambda out: self.enc_attn_net(out, source_hidden, source_mask)
        else:
            enc_attn_fn = None

        output = dict()
        if self.training:
            # due to the teacher forcing, the parallel mode is OK for training
            tree_inp = derivation_tree[:, :, :-1]  # excluding <RuleEnd>
            # token_fidelity does not come with an indicator for LHS
            inp_is_nt = token_fidelity[:, :, :-1] == self.nt_fi  # excluding <RuleEnd>

            gold_labels = self._get_gold_label(derivation_tree, token_fidelity)
            mask = gold_labels[2]
            # A tuple of the three
            # is_nt: (batch, derivation, rhs_seq - 1)
            # nt_logits: (batch, derivation, rhs_seq - 1, #NT)
            # t_logits: (batch, derivation, rhs_seq - 1, #T)
            pda_logits = self.ebnf_npda(tree_inp, inp_is_nt, attn_fn=enc_attn_fn, parallel_mode=True, inp_mask=mask)
            predictions = self._get_prediction(pda_logits)
            loss = self._compute_loss(self._get_gold_label(derivation_tree, token_fidelity),
                                      pda_logits, predictions)
            output['loss'] = loss

        elif self.inference_with_oracle_tree:
            # with oracle trees, the derivation number and LHS is given.
            tree_inp = derivation_tree[:, :, :2]  # the oracle tree, with LHS
            expansion_len = None if derivation_tree is None else derivation_tree.size()[-1] - 2
            pda_logits = self.ebnf_npda(tree_inp, None,
                                        max_expansion_len=expansion_len,
                                        attn_fn=enc_attn_fn, parallel_mode=True)
            predictions = self._get_prediction(pda_logits)

        else:
            derivation_count = None if derivation_tree is None else derivation_tree.size()[1]
            expansion_len = None if derivation_tree is None else derivation_tree.size()[-1] - 2
            raise NotImplementedError

        # anything in preds: (batch, derivation, rhs_seq - 1)
        output['preds'] = predictions

        if derivation_tree is not None:
            analysis = self._compute_metrics_and_analyze_errors(
                self._get_gold_label(derivation_tree, token_fidelity),
                pda_logits, predictions
            )
            analysis['all_lhs'] = derivation_tree[:, :, 0]
            analysis['source_tokens'] = source_tokens
            output['deliberate_analysis'] = analysis

        output.update(source_tokens=source_tokens, derivation_tree=derivation_tree, token_fidelity=token_fidelity)
        return output

    def _get_prediction(self,
                        pda_output: Tuple3FloatTensor,
                        ) -> Tuple3LongTensor:
        is_nt_prob, nt_logits, t_logits = pda_output
        predictions: Tuple3LongTensor = (is_nt_prob > 0.5, nt_logits.argmax(dim=-1), t_logits.argmax(dim=-1))
        return predictions

    def _get_gold_label(self,
                        derivation_tree: torch.LongTensor,
                        token_fidelity: torch.LongTensor,
                        ) -> Tuple5LongTensor:
        tree_out = derivation_tree[:, :, 2:]  # excluding LHS and <RuleGo>
        out_is_nt = token_fidelity[:, :, 1:] == self.nt_fi  # excluding <RuleGo>
        out_is_t = out_is_nt.logical_not()

        # padding is a terminal, padding_id for a non-terminal place is meaningless by convention
        # mask: (batch, derivation, rhs_seq - 1), mask is 1 for any non-padded token.
        mask = ((tree_out == self.tok_pad) * out_is_nt).logical_not()

        # gold token id is set to 0 if the position doesn't match the correct symbol type
        safe_nt_out = tree_out * out_is_nt
        safe_t_out = tree_out * out_is_t

        return out_is_nt, out_is_t, mask, safe_nt_out, safe_t_out

    def _compute_metrics_and_analyze_errors(self,
                                            gold_labels: Tuple5LongTensor,
                                            pda_logits: Tuple3FloatTensor,
                                            preds: Tuple3LongTensor,
                                            ) -> dict:
        out_is_nt, out_is_t, mask, safe_nt_out, safe_t_out = gold_labels

        # metric computation - PPL
        xent = self._get_batch_xent(pda_logits,
                                    (out_is_nt, safe_nt_out, safe_t_out),
                                    (mask, mask * out_is_nt, mask * out_is_t))
        for instance_xent in sum(xent):
            self.ppl(instance_xent.sum())

        # metric computation - Acc
        # *err: (batch, derivation, rhs_seq - 1)
        is_nt_err, nt_err, t_err = self._get_errors(gold_labels, preds)
        total_err = (is_nt_err + nt_err + t_err).sum(dim=(1, 2)) > 0
        for instance_err in total_err:
            self.err(instance_err)

        deliberate_analysis = {
            "error": [is_nt_err, nt_err, t_err],
            "gold": [mask, out_is_nt, safe_nt_out, safe_t_out],
        }
        return deliberate_analysis

    def _compute_loss(self, gold_labels, pda_output, preds):
        out_is_nt, out_is_t, mask, safe_nt_out, safe_t_out = gold_labels
        # loss_mask: (batch, derivation, 1)
        derivation_loss_mask = self._get_derivation_loss_mask(gold_labels, preds)
        # loss_mask: (batch, derivation, rhs_seq - 1)
        loss_mask = derivation_loss_mask * mask
        xent = self._get_batch_xent(pda_logits=pda_output,
                                    pda_gold_labels=(out_is_nt, safe_nt_out, safe_t_out),
                                    pda_weights=(loss_mask, loss_mask * out_is_nt, loss_mask * out_is_t),
                                    )
        loss = sum(xent).mean(0).sum()
        return loss

    def _get_derivation_loss_mask(self, gold_labels: Tuple5Tensor, preds: Tuple3LongTensor) -> torch.Tensor:
        errors = self._get_errors(gold_labels, preds)
        total_err = sum(errors)
        # derivation_loss_mask: (batch, derivation, 1)
        derivation_loss_mask = (total_err.sum(-1).cumsum(dim=-1) > 0).unsqueeze(-1)

        return derivation_loss_mask

    def _get_errors(self, gold_labels: Tuple5Tensor, preds: Tuple3LongTensor) -> Tuple3LongTensor:
        # total_error, and anything in preds or gold_labels: (batch, derivation, rhs_seq - 1)
        # predictions = (is_nt_prob > 0.5, nt_logits.argmax(dim=-1), t_logits.argmax(dim=-1))
        out_is_nt, out_is_t, mask, safe_nt_out, safe_t_out = gold_labels

        is_nt_err = (preds[0] != out_is_nt) * mask
        nt_err = (preds[1] != safe_nt_out) * mask * out_is_nt
        t_err = (preds[2] != safe_t_out) * mask * out_is_t
        return is_nt_err, nt_err, t_err


    def _get_batch_xent(self,
                        pda_logits: Tuple3FloatTensor,
                        pda_gold_labels: Tuple3LongTensor,
                        pda_weights: Tuple3Tensor,
                        ) -> Tuple3FloatTensor:
        # is_nt: (batch, derivation, rhs_seq - 1)
        # nt_logits: (batch, derivation, rhs_seq - 1, #NT)
        # t_logits: (batch, derivation, rhs_seq - 1, #T)
        is_nt_prob, nt_logits, t_logits = pda_logits

        # all others: (batch, derivation, rhs_seq - 1)
        out_is_nt, safe_nt_out, safe_t_out = pda_gold_labels
        is_nt_mask, nt_mask, t_mask = pda_weights

        is_nt_xent = binary_cross_entropy(is_nt_prob, out_is_nt.float()) * is_nt_mask
        xent_is_nt = is_nt_xent.sum([1, 2]) / (is_nt_mask.sum([1, 2]) + 1e-13)
        xent_nt = seq_cross_ent(nt_logits, safe_nt_out, nt_mask.float(), average=None)
        xent_t = seq_cross_ent(t_logits, safe_t_out, t_mask.float(), average=None)
        return xent_is_nt, xent_nt, xent_t



