import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from utils.nn import seq_cross_ent, seq_likelihood
from .ebnf_npda import NeuralEBNF
from allennlp.training.metrics.perplexity import Perplexity, Average


class EBNFTreeLM(nn.Module):
    def __init__(self, ebnf_npda: NeuralEBNF, tok_pad_id: int, nt_fi: int):
        super().__init__()
        self.ppl = Perplexity()
        self.err = Average()
        self.ebnf_pda = ebnf_npda
        self.tok_pad = tok_pad_id
        self.nt_fi = nt_fi

    def get_metric(self, reset=False):
        ppl = self.ppl.get_metric(reset)
        err = self.err.get_metric(reset)
        return {"PPL": ppl, "ERR": err}

    def forward(self, derivation_tree: torch.LongTensor, token_fidelity: torch.LongTensor):
        """
        :param derivation_tree: (batch, derivation, seq), seq: [lhs, *rhs]
        :param token_fidelity: (batch, derivation, rhs_seq), rhs_seq = seq - 1, rhs: [<RuleGo> ... <RuleEnd>]
        :return:
        """
        tree_inp = derivation_tree[:, :, :-1]   # excluding <RuleEnd>
        tree_out = derivation_tree[:, :, 2:]    # excluding LHS and <RuleGo>

        # token_fidelity does not come with an indicator for LHS
        inp_is_nt = token_fidelity[:, :, :-1] == self.nt_fi # excluding <RuleEnd>
        out_is_nt = token_fidelity[:, :, 1:] == self.nt_fi  # excluding <RuleGo>
        out_is_t = out_is_nt.logical_not()

        # padding is a terminal, padding_id for a non-terminal place is meaningless by convention
        # mask: (batch, derivation, rhs_seq - 1), mask is 1 for any non-padded token.
        mask = ((tree_out == self.tok_pad) * out_is_nt).logical_not()
        # is_nt: (batch, derivation, rhs_seq - 1)
        # nt_logits: (batch, derivation, rhs_seq - 1, #NT)
        # t_logits: (batch, derivation, rhs_seq - 1, #T)
        is_nt_prob, nt_logits, t_logits = self.ebnf_pda(tree_inp, inp_is_nt, parallel_mode=True)

        # gold token id is set to 0 if the position doesn't match the correct symbol type
        safe_nt_out = tree_out * out_is_nt
        safe_t_out = tree_out * out_is_t

        output = {}
        is_nt_xent = binary_cross_entropy(is_nt_prob, out_is_nt.float()) * mask
        reducible_dims = list(range(is_nt_xent.ndim)[1:])
        loss_is_nt = is_nt_xent.sum(reducible_dims) / (mask.sum(reducible_dims) + 1e-13)
        loss_nt = seq_cross_ent(nt_logits, safe_nt_out, mask * out_is_nt, average=None)
        loss_t = seq_cross_ent(t_logits, safe_t_out, mask * out_is_t, average=None)
        loss = (loss_is_nt + loss_nt + loss_t).mean()
        if self.training:
            output['loss'] = loss

        # metric computation - PPL
        for instance_logprobs in zip(loss_is_nt, loss_nt, loss_t):
            self.ppl(sum(instance_logprobs))
        # metric computation - Acc
        # anything in preds: (batch, derivation, rhs_seq - 1)
        preds = (is_nt_prob > 0.5, nt_logits.argmax(dim=-1), t_logits.argmax(dim=-1))
        # *err: (batch, derivation, rhs_seq - 1)
        is_nt_err = (preds[0] != out_is_nt) * mask
        nt_err = (preds[1] != safe_nt_out) * mask * out_is_nt
        t_err = (preds[2] != safe_t_out) * mask * out_is_t
        total_err = (is_nt_err + nt_err + t_err).sum(dim=(1, 2)) > 0
        for instance_err in total_err:
            self.err(instance_err)
        output['preds'] = preds
        output['deliberate_analysis'] = {
            "error": [is_nt_err, nt_err, t_err],
            "gold": [mask, out_is_nt, safe_nt_out, safe_t_out],
            "all_lhs": derivation_tree[:, :, 0],
        }
        return output
