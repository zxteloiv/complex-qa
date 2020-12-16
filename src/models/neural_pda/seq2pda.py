from typing import Optional
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from utils.nn import seq_cross_ent, seq_likelihood
from models.modules.stacked_encoder import StackedEncoder
from .ebnf_npda import NeuralEBNF
from allennlp.training.metrics.perplexity import Perplexity, Average
from utils.nn import prepare_input_mask

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

        output = {}
        if self.training:
            # due to the teacher forcing, the parallel mode is OK for training
            tree_inp = derivation_tree[:, :, :-1]  # excluding <RuleEnd>
            # token_fidelity does not come with an indicator for LHS
            inp_is_nt = token_fidelity[:, :, :-1] == self.nt_fi  # excluding <RuleEnd>

            gold_labels = self._get_gold_label(derivation_tree, token_fidelity)
            mask = gold_labels[3]
            # A tuple of the three
            # is_nt: (batch, derivation, rhs_seq - 1)
            # nt_logits: (batch, derivation, rhs_seq - 1, #NT)
            # t_logits: (batch, derivation, rhs_seq - 1, #T)
            pda_logits = self.ebnf_npda(tree_inp, inp_is_nt, attn_fn=enc_attn_fn, parallel_mode=True, inp_mask=mask)
            loss_is_nt, loss_nt, loss_t = self._get_batch_xent(gold_labels, pda_logits)
            loss = (loss_is_nt + loss_nt + loss_t).mean()
            output['loss'] = loss

        elif self.inference_with_oracle_tree:
            # with oracle trees, the derivation number and LHS is given.
            tree_inp = derivation_tree[:, :, :2]  # the oracle tree, with LHS
            expansion_len = None if derivation_tree is None else derivation_tree.size()[-1] - 2
            pda_logits = self.ebnf_npda(tree_inp, None,
                                        max_expansion_len=expansion_len,
                                        attn_fn=enc_attn_fn, parallel_mode=True)

        else:
            derivation_count = None if derivation_tree is None else derivation_tree.size()[1]
            expansion_len = None if derivation_tree is None else derivation_tree.size()[-1] - 2
            raise NotImplementedError

        # anything in preds: (batch, derivation, rhs_seq - 1)
        is_nt_prob, nt_logits, t_logits = pda_logits
        predictions = (is_nt_prob > 0.5, nt_logits.argmax(dim=-1), t_logits.argmax(dim=-1))
        if derivation_tree is not None:
            # for clarity, the gold labels and xents inside metric computation will be duplicated during training
            gold_labels = self._get_gold_label(derivation_tree, token_fidelity)
            analysis = self._compute_metrics(gold_labels, pda_logits, predictions)
            analysis['all_lhs'] = derivation_tree[:, :, 0]
            analysis['source_tokens'] = source_tokens
            output['deliberate_analysis'] = analysis

        output['preds'] = predictions
        output.update(source_tokens=source_tokens, derivation_tree=derivation_tree, token_fidelity=token_fidelity)
        return output

    def _compute_metrics(self, gold_labels, pda_logits, preds):
        xent_is_nt, xent_nt, xent_t = self._get_batch_xent(gold_labels, pda_logits)
        tree_out, out_is_nt, out_is_t, mask, safe_nt_out, safe_t_out = gold_labels

        # metric computation - PPL
        for instance_logprobs in zip(xent_is_nt, xent_nt, xent_t):
            self.ppl(sum(instance_logprobs))
        # metric computation - Acc
        # *err: (batch, derivation, rhs_seq - 1)
        is_nt_err = (preds[0] != out_is_nt) * mask
        nt_err = (preds[1] != safe_nt_out) * mask * out_is_nt
        t_err = (preds[2] != safe_t_out) * mask * out_is_t
        total_err = (is_nt_err + nt_err + t_err).sum(dim=(1, 2)) > 0
        for instance_err in total_err:
            self.err(instance_err)
        deliberate_analysis = {
            "error": [is_nt_err, nt_err, t_err],
            "gold": [mask, out_is_nt, safe_nt_out, safe_t_out],
        }
        return deliberate_analysis

    def _get_batch_xent(self, gold_labels, pda_output):
        is_nt_prob, nt_logits, t_logits = pda_output
        # loss computation
        tree_out, out_is_nt, out_is_t, mask, safe_nt_out, safe_t_out = gold_labels
        is_nt_xent = binary_cross_entropy(is_nt_prob, out_is_nt.float()) * mask
        reducible_dims = list(range(is_nt_xent.ndim)[1:])
        xent_is_nt = is_nt_xent.sum(reducible_dims) / (mask.sum(reducible_dims) + 1e-13)
        xent_nt = seq_cross_ent(nt_logits, safe_nt_out, mask * out_is_nt, average=None)
        xent_t = seq_cross_ent(t_logits, safe_t_out, mask * out_is_t, average=None)
        return xent_is_nt, xent_nt, xent_t

    def _get_gold_label(self, derivation_tree, token_fidelity):
        tree_out = derivation_tree[:, :, 2:]  # excluding LHS and <RuleGo>
        out_is_nt = token_fidelity[:, :, 1:] == self.nt_fi  # excluding <RuleGo>
        out_is_t = out_is_nt.logical_not()

        # padding is a terminal, padding_id for a non-terminal place is meaningless by convention
        # mask: (batch, derivation, rhs_seq - 1), mask is 1 for any non-padded token.
        mask = ((tree_out == self.tok_pad) * out_is_nt).logical_not()

        # gold token id is set to 0 if the position doesn't match the correct symbol type
        safe_nt_out = tree_out * out_is_nt
        safe_t_out = tree_out * out_is_t

        return tree_out, out_is_nt, out_is_t, mask, safe_nt_out, safe_t_out



