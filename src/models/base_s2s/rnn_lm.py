from typing import Optional, List
import torch.nn
from ..modules.stacked_rnn_cell import StackedRNNCell
from allennlp.nn.util import masked_softmax
from allennlp.training.metrics.perplexity import Perplexity, Average
from utils.nn import seq_cross_ent, seq_likelihood

class RNNModel(torch.nn.Module):
    def __init__(self,
                 embedding,
                 rnn: StackedRNNCell,
                 prediction,
                 padding: int = 0,
                 attention: Optional = None,
                 attention_composer: Optional = None,
                 ):
        super().__init__()
        self.embedding = embedding
        self.rnn = rnn
        self.padding = padding
        self.prediction = prediction
        self.attention = attention
        self.composer = attention_composer

        self.ppl = Perplexity()
        self.err = Average()

    def forward(self, tok_seq: torch.Tensor, hx=None):
        """

        :param tok_seq: (batch, max_seq_len)
        :param hx: hidden states for stacked RNNCell
        :return:
        """
        inp = tok_seq[:, :-1].contiguous()
        tgt = tok_seq[:, 1:].contiguous()
        mask = (tgt != self.padding).long()

        seq = self.embedding(inp)
        logits = self.forward_emb(seq, mask, hx)

        xent, preds = self._compute_err(logits, tgt, mask)
        output = {"likelihoods": seq_likelihood(logits, tgt, mask), "preds": preds}
        if self.training:
            output['loss'] = xent

        return output

    def get_hx_with_initial_state(self, init_state: List[torch.Tensor]):
        hx, out = self.rnn.init_hidden_states(init_state)
        return hx

    def _compute_err(self, logits, targets, mask):
        xent = seq_cross_ent(logits, targets, mask)
        self.ppl(xent)

        preds = logits.argmax(dim=-1)
        total_err = ((preds != targets) * mask).sum(list(range(mask.ndim))[1:]) > 0
        for instance_err in total_err:
            self.err(instance_err)

        return xent, preds

    def forward_emb(self, seq, mask, hx):
        out_list = []
        for step in range(seq.size()[1]):
            step_emb = seq[:, step]
            hx, out = self.rnn(step_emb, hx)
            out_list.append(out)

        # out: (batch, len, hid)
        out = torch.stack(out_list, dim=1)

        if self.attention is None:
            pred_input = out

        else:
            # attn: (batch, len, len)
            attn = self.attention(out, out)
            weight = masked_softmax(attn, mask.unsqueeze(2), dim=1)
            ctx = weight.transpose(1, 2).matmul(out)
            pred_input = self.composer(ctx, out)

        logits = self.prediction(pred_input)
        return logits



