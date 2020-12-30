from typing import Optional, Tuple, Generic, List
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from utils.nn import seq_cross_ent, seq_likelihood
from models.modules.stacked_encoder import StackedEncoder
from allennlp.training.metrics.perplexity import Perplexity, Average
from utils.nn import prepare_input_mask
from .npda import NeuralPDA
from utils.seq_collector import SeqCollector
from utils.text_tool import make_human_readable_text
import logging

# from .tensor_typing_util import ( T3T, T3L, T3F, T4T, T4L, T4F, T5T, T5L, T5F, )
from .tensor_typing_util import *

class Seq2PDA(nn.Module):
    def __init__(self,
                 # modules
                 encoder: StackedEncoder,
                 src_embedding: nn.Embedding,
                 enc_attn_net: nn.Module,
                 npda: NeuralPDA,

                 # configuration
                 max_expansion_len: int,
                 src_ns: str,
                 tgt_ns: List[str],
                 vocab,
                 ):
        super().__init__()
        self.encoder = encoder
        self.src_embedder = src_embedding
        self.enc_attn_net = enc_attn_net
        self.npda = npda

        self.ppl = Perplexity()
        self.err = Average()
        self.tok_pad = 0
        self.max_expansion_len = max_expansion_len
        self.src_ns = src_ns
        self.tgt_ns = tgt_ns
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vocab = vocab

    def get_metric(self, reset=False):
        ppl = self.ppl.get_metric(reset)
        err = self.err.get_metric(reset)
        return {"PPL": ppl, "ERR": err}

    def forward(self, *args, **kwargs):
        if self.training:
            return self._forward_amortized_training(*args, **kwargs)

        else:
            return self._forward_inference(*args, **kwargs)

    def _forward_amortized_training(self,
                                    source_tokens: LT,
                                    rhs_symbols: LT,
                                    parental_growth: LT,
                                    fraternal_growth: LT,
                                    rhs_exact_symbols: LT,
                                    target_tokens: LT,
                                    ):
        """
        :param source_tokens: (batch, max_src_len)
        :param rhs_symbols: (batch, max_tgt_len), separated by every max_derivation_symbols
        :param parental_growth: (batch, max_tgt_len)
        :param fraternal_growth: (batch, max_tgt_len)
        :param rhs_exact_symbols: (batch, max_tgt_len)
        :return:
        """
        enc_attn_fn = self._encode_source(source_tokens)
        self.npda.init_automata(source_tokens.size()[0], source_tokens.device, enc_attn_fn)

        step = loss = 0
        exact_token_list = []
        while self.npda.continue_derivation() and step * self.max_expansion_len < source_tokens.size()[1]:
            rhs_p, lhs_mask = self.npda()
            rhs_gold = self._get_step_gold(step, rhs_symbols, parental_growth, fraternal_growth, rhs_exact_symbols)

            derivation_loss = self._compute_loss(rhs_p, rhs_gold, lhs_mask)
            derivation_loss.backward(retain_graph=True)
            loss = loss + derivation_loss.detach()
            step += 1

            exact_token_list.append(self._arrange_target_tokens(self.npda.inference(rhs_p), lhs_mask))
            self.npda.push_predictions_onto_stack(rhs_gold, lhs_mask)

        # compute metrics
        if len(exact_token_list) > 0:
            exact_tokens = torch.cat(exact_token_list, dim=-1)
            self._compute_err(exact_tokens, target_tokens)
        else:
            self.logger.warning('No exact token generated, which is impossible.')
        self.ppl(loss)

        output = {'loss': loss}
        self.npda.reset_automata()
        return output

    def _encode_source(self, source_tokens):
        source_tokens, source_mask = prepare_input_mask(source_tokens, padding_val=self.tok_pad)
        source = self.src_embedder(source_tokens)
        source_hidden, _ = self.encoder(source, source_mask)
        enc_attn_fn = lambda out: self.enc_attn_net(out, source_hidden, source_mask)
        return enc_attn_fn

    def _get_step_gold(self, step_id: int, *args):
        step_size = self.max_expansion_len
        step_starts = step_id * step_size
        step_ends = step_starts + step_size
        clip = lambda t: t[:, step_starts:step_ends]
        return tuple(map(clip, args))

    def _compute_loss(self, rhs_p, rhs_gold, lhs_mask):
        # lhs_mask: (batch, 1)
        lhs_mask = lhs_mask.unsqueeze(-1)
        fraternal_p = rhs_p[2]
        f_mask = rhs_gold[2]
        comp_len = min(fraternal_p.size()[-1], f_mask.size()[-1])

        # mask: (batch, comp_len)
        mask = (lhs_mask * f_mask)[:, :comp_len]

        _clip_logprob = lambda t: (t + 1e-13).log()[:, :comp_len]

        # *symbol_logp: (batch, comp_len, V)
        # p/f*_p: (batch, comp_len)
        symbol_logp = _clip_logprob(rhs_p[0])
        exact_symbol_logp = _clip_logprob(rhs_p[3])
        parental_p = rhs_p[1][:, :comp_len]
        fraternal_p = fraternal_p[:, :comp_len]

        # gold_labels: (batch, comp_len)
        symbol, p_growth, f_mask, exact_symbol = map(lambda t: t[:, :comp_len], rhs_gold)

        # (batch, comp_len, V) -> (batch, comp_len, 1) -> (batch, comp_len)
        symbol_loss = -symbol_logp.gather(dim=-1, index=symbol.unsqueeze(-1)).squeeze(-1) * mask
        exact_symbol_loss = -exact_symbol_logp.gather(dim=-1, index=exact_symbol.unsqueeze(-1)).squeeze(-1) * mask
        parental_loss = binary_cross_entropy(parental_p, p_growth.float()) * mask
        fraternal_loss = binary_cross_entropy(fraternal_p, f_mask.float()) * mask

        loss = (symbol_loss + exact_symbol_loss + parental_loss + fraternal_loss).mean(0).sum()
        return loss


    def _arrange_target_tokens(self, predictions, lhs_mask):
        # (batch, pad_seq_len1)
        _, _, f_mask_hat, exact_symbol_hat = predictions
        target_hats = exact_symbol_hat * f_mask_hat * lhs_mask.unsqueeze(-1)
        return target_hats

    def _compute_err(self, batch_exact_tokens, batch_target_tokens):
        """
        :param batch_exact_tokens: (batch, seq1)
        :param batch_target_tokens: (batch, seq2)
        :return:
        """
        for instance_pair in zip(batch_exact_tokens, batch_target_tokens):
            exact_tokens, target_tokens = list(map(lambda t: list(filter(lambda x: x != self.tok_pad, t.tolist())), instance_pair))
            if len(exact_tokens) == len(target_tokens) and all(x == y for x, y in zip(exact_tokens, target_tokens)):
                self.err(1.)
            else:
                self.err(0.)

    def _forward_inference(self,
                           source_tokens: LT,
                           rhs_symbols: NullOrLT,
                           parental_growth: NullOrLT,
                           fraternal_growth: NullOrLT,
                           rhs_exact_symbols: NullOrLT,
                           target_tokens: NullOrLT,
                           ):
        enc_attn_fn = self._encode_source(source_tokens)
        self.npda.init_automata(source_tokens.size()[0], source_tokens.device, enc_attn_fn)

        predicted_tokens = [
            self._arrange_target_tokens(derivation_predictions, derivation_lhs_masks)
            for _, derivation_predictions, derivation_lhs_masks in zip(*self.npda())
        ]

        predicted_tokens = torch.cat(predicted_tokens, dim=-1)
        self._compute_err(predicted_tokens, target_tokens)

        output = {
            "source": source_tokens,
            "target": target_tokens,
            "prediction": predicted_tokens,
        }
        return output

    def make_human_readable_output(self, output):
        output['source_tokens'] = make_human_readable_text(output['source'], self.vocab, self.src_ns)
        output['target_tokens'] = make_human_readable_text(output['target'], self.vocab, self.tgt_ns[1])
        output['predicted_tokens'] = make_human_readable_text(output['prediction'], self.vocab, self.tgt_ns[1])
        return output



