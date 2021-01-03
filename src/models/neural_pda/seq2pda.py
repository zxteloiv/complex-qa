from typing import Optional, Tuple, Generic, List
import torch
import torch.nn as nn
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
                                    rhs_exact_tokens: LT,
                                    mask: LT,
                                    target_tokens: LT,
                                    ):
        """
        :param source_tokens: (batch, max_src_len)
        :param rhs_symbols: (batch, max_tgt_len), separated by every max_derivation_symbols
        :param parental_growth: (batch, max_tgt_len)
        :param fraternal_growth: (batch, max_tgt_len)
        :param rhs_exact_tokens: (batch, max_tgt_len)
        :return:
        """
        enc_attn_fn = self._encode_source(source_tokens)
        self.npda.init_automata(source_tokens.size()[0], source_tokens.device, enc_attn_fn)

        step = 0
        loss = valid_derivation_num = 0
        exact_token_list = []
        while self.npda.continue_derivation() and step * self.max_expansion_len < mask.size()[1]:
            derivation = self.npda()
            # lhs, lhs_mask, grammar_guide, opts_logp, topo_preds, exact_token_p = derivation
            gold = self._get_step_gold(step, rhs_symbols, parental_growth, fraternal_growth, rhs_exact_tokens, mask)
            derivation_loss = self._compute_loss(derivation, gold)
            derivation_loss.backward(retain_graph=True)
            loss = loss + derivation_loss.detach()
            step += 1

            # any valid derivation must expand the RHS starting with a START token, and the mask is set to 1.
            valid_derivation_num += gold[-1][:, 0]
            exact_token_list.append(self._arrange_target_tokens(derivation))
            self.npda.push_predictions_onto_stack(gold[:3], derivation[1])

        # compute metrics
        if len(exact_token_list) > 0:
            exact_tokens = torch.cat(exact_token_list, dim=-1)
            self._compute_err(exact_tokens, target_tokens)
        else:
            self.logger.warning('No exact token generated, which is impossible.')

        normalized_loss = loss / valid_derivation_num.float().mean()
        self.ppl(normalized_loss)
        output = {'loss': normalized_loss}
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

    def _compute_loss(self, derivation, gold):
        """
        :param derivation: Tuple of 6 objects
            lhs, lhs_mask: (batch,) ;
            grammar_guide: (batch, opt_num, 4, max_seq) ;
            opts_logp: (batch, opt_num) ;
            topo_preds: Tuple[(batch, max_seq) * 4] ;
            exact_token_p: [(batch, max_seq, V)]
        :param gold:
            symbol: (batch, max_derivation_len), separated by every max_derivation_symbols
            parental_growth: (batch, max_derivation_len)
            fraternal_growth: (batch, max_derivation_len)
            exact_token: (batch, max_derivation_len)
            mask: (batch, max_derivation_len)
        :return:
        """
        lhs, lhs_mask, grammar_guide, opts_logp, topo_preds, exact_token_p = derivation
        comp_len = grammar_guide.size()[-1]

        symbol, p_growth, _, exact_token, seq_mask = map(lambda t: t[:, :comp_len], gold)

        et_mask = seq_mask * (p_growth == 0)
        et_logp = (exact_token_p + 1e-13).log()
        et_loss = -et_logp.gather(dim=-1, index=exact_token.unsqueeze(-1)).squeeze(-1) * et_mask
        per_batch_loss = et_loss.sum(1) / (et_mask.sum(1) + 1e-13)
        num_non_empty_sequences = ((et_mask.sum(1) > 0).float().sum() + 1e-13)
        token_loss = per_batch_loss.sum() / num_non_empty_sequences

        # seq_symbol_opts: (batch, opt_num, comp_len)
        seq_symbol_opts = grammar_guide[:, :, 0, :]

        # seq_opt_weights: (batch, opt_num, comp_len)
        seq_opt_weights = (symbol.unsqueeze(1) == seq_symbol_opts)   # target seq is different with the option
        seq_opt_weights = seq_opt_weights * seq_mask.unsqueeze(1)    # conducting comparisons only at valid locations

        # opt_weights: (batch, opt_num), only the correct opt is marked as 1
        opt_weights = seq_opt_weights.sum(-1) == seq_mask.sum(dim=-1).unsqueeze(1)
        batch_mask = seq_mask.sum(dim=-1) > 0
        topology_loss_per_batch = -(opt_weights * opts_logp).sum(-1) * batch_mask

        # any valid derivation has a starting symbol by default and thus must be a non-empty sequence
        topology_loss = topology_loss_per_batch.sum() / num_non_empty_sequences

        return token_loss + topology_loss

    def _arrange_target_tokens(self, derivation):
        lhs, lhs_mask, grammar_guide, opts_logp, topo_preds, exact_token_p = derivation
        p_growth, mask = topo_preds[1], topo_preds[3]
        # target_hats: (batch, seq)
        target_hats = exact_token_p.argmax(dim=-1) * p_growth.logical_not() * mask * lhs_mask.unsqueeze(-1)
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
                self.err(0.)
            else:
                self.err(1.)

    def _forward_inference(self,
                           source_tokens: LT,
                           rhs_symbols: NullOrLT,
                           parental_growth: NullOrLT,
                           fraternal_growth: NullOrLT,
                           rhs_exact_tokens: NullOrLT,
                           mask: LT,
                           target_tokens: NullOrLT,
                           ):
        enc_attn_fn = self._encode_source(source_tokens)
        self.npda.init_automata(source_tokens.size()[0], source_tokens.device, enc_attn_fn)

        step = 0
        exact_token_list = []
        while self.npda.continue_derivation():
            derivation = self.npda()
            lhs_mask, topo_preds = derivation[1], derivation[-2]
            step += 1
            exact_token_list.append(self._arrange_target_tokens(derivation))
            self.npda.push_predictions_onto_stack(topo_preds[:3], derivation[1])

        # compute metrics
        assert len(exact_token_list) > 0
        exact_tokens = torch.cat(exact_token_list, dim=-1)
        self._compute_err(exact_tokens, target_tokens)
        output = {
            "source": source_tokens,
            "target": target_tokens,
            "prediction": exact_tokens,
        }
        return output

    def make_human_readable_output(self, output):
        output['source_tokens'] = make_human_readable_text(output['source'], self.vocab, self.src_ns)
        output['target_tokens'] = make_human_readable_text(output['target'], self.vocab, self.tgt_ns[1])
        output['predicted_tokens'] = make_human_readable_text(output['prediction'], self.vocab, self.tgt_ns[1])
        return output



