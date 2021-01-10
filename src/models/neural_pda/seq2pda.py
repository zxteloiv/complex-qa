from typing import Optional, Tuple, Generic, List
import torch
import torch.nn as nn
from models.modules.stacked_encoder import StackedEncoder
from allennlp.training.metrics.perplexity import Perplexity, Average
from utils.nn import prepare_input_mask, seq_cross_ent, init_state_for_stacked_rnn
from .npda import NeuralPDA
from utils.seq_collector import SeqCollector
from utils.text_tool import make_human_readable_text
from .batched_stack import TensorBatchStack
from ..base_s2s.rnn_lm import RNNModel
import logging
import gc

# from .tensor_typing_util import ( T3T, T3L, T3F, T4T, T4L, T4F, T5T, T5L, T5F, )
from .tensor_typing_util import *


def _efficiently_optimize(loss_list, optim):
    filtered_loss = list(filter(lambda l: l > 1e-13, loss_list))
    loss = 0
    if len(filtered_loss) > 0:
        for l in filtered_loss:
            loss = loss + l
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
    return loss.detach() if isinstance(loss, torch.Tensor) else loss


class Seq2PDA(nn.Module):
    def __init__(self,
                 # modules
                 encoder: StackedEncoder,
                 src_embedding: nn.Embedding,
                 enc_attn_net: nn.Module,
                 npda: NeuralPDA,

                 src_lm: RNNModel,

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
        self.lm = src_lm

        self.loss = Average()
        self.err = Average()
        self.tok_pad = 0
        self.max_expansion_len = max_expansion_len
        self.src_ns = src_ns
        self.tgt_ns = tgt_ns
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vocab = vocab

    def get_metric(self, reset=False):
        loss = self.loss.get_metric(reset)
        err = self.err.get_metric(reset)
        return {"Amortized_loss": loss, "ERR": err}

    def forward(self, *args, **kwargs):
        if self.training:
            return self._forward_amortized_training(*args, **kwargs)

        else:
            return self._forward_inference(*args, **kwargs)

    def _forward_amortized_training(self,
                                    source_tokens: LT,
                                    rhs_symbols: LT,
                                    parental_growth: LT,
                                    rhs_exact_tokens: LT,
                                    mask: LT,
                                    target_tokens: LT,
                                    optim = None,
                                    ):
        """
        :param source_tokens: (batch, max_src_len)
        :param rhs_symbols: (batch, max_tgt_len), separated by every max_derivation_symbols
        :param parental_growth: (batch, max_tgt_len)
        :param rhs_exact_tokens: (batch, max_tgt_len)
        :return:
        """
        enc_attn_fn = self._encode_source(source_tokens)
        self.npda.init_automata(source_tokens.size()[0], source_tokens.device, enc_attn_fn)

        step = 0
        separate_loss = []
        derivation_loss = []
        valid_derivation_num = 0
        token_stack = TensorBatchStack(source_tokens.size()[0], 1000,
                                       item_size=1, dtype=torch.long, device=source_tokens.device)
        while self.npda.continue_derivation() and step * self.max_expansion_len < mask.size()[1]:
            (full_step_symbols,) = self._get_step_gold(step, self.max_expansion_len, rhs_symbols)
            lhs, lhs_mask, grammar_guide, opt_prob, exact_token_logit = self.npda(step_symbols=full_step_symbols)

            src_hx = self.lm.get_hx_with_initial_state(
                init_state_for_stacked_rnn([self.npda.tree_state], self.lm.rnn.get_layer_num(), "all")
            )

            z2x_loss = self.lm(source_tokens, src_hx)['loss']

            step_symbols, step_p_growth, step_exact_tokens, step_mask = self._get_step_gold(
                step, grammar_guide.size()[-1], rhs_symbols, parental_growth, rhs_exact_tokens, mask
            )

            token_loss, topo_loss = self._compute_loss(grammar_guide, opt_prob, exact_token_logit,
                                                       step_symbols, step_p_growth, step_exact_tokens, step_mask,)
            derivation_loss.append(_efficiently_optimize([z2x_loss, token_loss, topo_loss], optim))

            separate_loss.append((z2x_loss.item(), token_loss.item(), topo_loss.item()))

            step += 1
            # any valid derivation must expand the RHS starting with a START token, and the mask is set to 1.
            valid_derivation_num += (step_mask.sum(-1) > 0).float().mean()
            self.npda.push_predictions_onto_stack(step_symbols, step_p_growth, step_mask, lhs_mask=None)

            predicted_symbol, predicted_p_growth, predicted_mask = self._infer_topology_greedily(opt_prob, grammar_guide)
            self._arrange_predicted_tokens(token_stack, exact_token_logit, lhs_mask, predicted_p_growth, predicted_mask)
            self.npda.stop_gradient_for_tree_state()

        # print(separate_loss)
        print(list(sum(l) / (valid_derivation_num + 1e-15) for l in zip(*separate_loss)))
        normalized_loss = sum(derivation_loss) / (valid_derivation_num + 1e-15)
        self.loss(normalized_loss)
        output = {'loss': normalized_loss}

        # compute metrics
        self._compute_err(token_stack, target_tokens)
        self.npda.reset_automata()
        return output

    def _encode_source(self, source_tokens):
        source_tokens, source_mask = prepare_input_mask(source_tokens, padding_val=self.tok_pad)
        source = self.src_embedder(source_tokens)
        source_hidden, _ = self.encoder(source, source_mask)
        enc_attn_fn = lambda out: self.enc_attn_net(out, source_hidden, source_mask)
        return enc_attn_fn

    def _get_step_gold(self, step_id: int, compatible_len: int, *args):
        max_size = self.max_expansion_len
        step_starts = step_id * max_size
        step_ends = step_starts + min(compatible_len, max_size)
        clip = lambda t: t[:, step_starts:step_ends]
        return tuple(map(clip, args))

    def _compute_loss(self,
                      grammar_guide, opt_prob, exact_token_logit,   # derivation output
                      step_symbols, step_p_growth, step_exact_tokens, step_mask,    # clipped gold target
                      ):
        """
        :param grammar_guide: (batch, opt_num, 4, max_seq)
        :param opt_prob: (batch, opt_num)
        :param exact_token_logit: [(batch, max_seq, V)]
        :param step_symbols: (batch, compatible_len), separated by every max_derivation_symbols
        :param step_p_growth: (batch, compatible_len)
        :param step_exact_tokens: (batch, compatible_len)
        :param step_mask: (batch, compatible_len)
        :return:
        """
        # symbol, p_growth, exact_token, seq_mask = map(lambda t: t[:, :comp_len], gold)

        token_loss = seq_cross_ent(exact_token_logit,
                                   step_exact_tokens,
                                   (step_mask * (step_p_growth == 0)).float(),
                                   average="batch")

        # gold_choice: (batch, 1)
        gold_choice = self.npda.find_choice_by_symbols(step_symbols, grammar_guide).unsqueeze(-1)
        non_empty_seq = step_mask.sum(dim=-1) > 0                           # non_empty_seq: (batch,)
        opt_logp = (opt_prob + 1e-15).log()                                 # opt_logp: (batch, opt_num)
        topology_batch_loss = -opt_logp.gather(dim=-1, index=gold_choice)   # topo_batch_loss: (batch, 1)
        topology_loss = (topology_batch_loss.squeeze(-1) * non_empty_seq).sum() / (non_empty_seq.sum() + 1e-15)
        return token_loss, topology_loss

    def _arrange_predicted_tokens(self, stack: TensorBatchStack,
                                  exact_token_logit, lhs_mask,
                                  predicted_p_growth, predicted_mask,
                                  ):
        # predicted_tokens: (batch, seq)
        predicted_tokens = exact_token_logit.argmax(dim=-1)

        for step in range(predicted_tokens.size()[-1]):
            push_mask = (predicted_p_growth[:, step] == 0) * predicted_mask[:, step] * lhs_mask
            stack.push(predicted_tokens[:, step].unsqueeze(-1), push_mask.long())

    def _arrange_predicted_symbols(self, stack: TensorBatchStack, lhs_mask, predicted_symbols, predicted_mask):
        # predicted_symbols: (batch, seq)
        for step in range(predicted_symbols.size()[-1]):
            push_mask = predicted_mask[:, step] * lhs_mask
            stack.push(predicted_symbols[:, step].unsqueeze(-1), push_mask.long())

    def _infer_topology_greedily(self, opt_prob, grammar_guide):
        greedy_choice = opt_prob.argmax(dim=-1)
        (
            predicted_symbol, predicted_p_growth, predicted_mask
        ) = self.npda.retrieve_the_chosen_structure(greedy_choice, grammar_guide)
        return predicted_symbol, predicted_p_growth, predicted_mask

    def _compute_err(self, exact_tokens_stack: TensorBatchStack, batch_target_tokens):
        """
        :param exact_tokens_stack: the stack
        :param batch_target_tokens: (batch, seq2)
        :return:
        """
        # pred_token: (batch, max_len, 1), the last dimension is the symbol length, and is always 1 in our case
        # pred_mask: (batch, max_len)
        pred_tokens, pred_masks = exact_tokens_stack.dump()

        pred_tokens = pred_tokens.squeeze(-1)
        gold_tokens, gold_masks = prepare_input_mask(batch_target_tokens, padding_val=self.tok_pad)

        for p_tok, p_m, g_tok, g_m in zip(pred_tokens, pred_masks, gold_tokens, gold_masks):
            if p_m.sum() != g_m.sum():
                self.err(1.)
                continue

            min_len = min(p_m.size()[-1], g_m.size()[-1])
            if ((p_tok * p_m)[:min_len] == (g_tok * g_m)[:min_len]).all():
                self.err(0.)

            else:
                self.err(1.)

    def _forward_inference(self,
                           source_tokens: LT,
                           rhs_symbols: NullOrLT,
                           parental_growth: NullOrLT,
                           rhs_exact_tokens: NullOrLT,
                           mask: LT,
                           target_tokens: NullOrLT,
                           ):
        enc_attn_fn = self._encode_source(source_tokens)
        self.npda.init_automata(source_tokens.size()[0], source_tokens.device, enc_attn_fn)

        batch_sz = source_tokens.size()[0]
        token_stack = TensorBatchStack(batch_sz, 1000, item_size=1, dtype=torch.long, device=source_tokens.device)
        symbol_stack = TensorBatchStack(batch_sz, 5000, item_size=1, dtype=torch.long, device=source_tokens.device)
        while self.npda.continue_derivation():
            lhs, lhs_mask, grammar_guide, opt_prob, exact_token_logit = self.npda()

            predicted_symbol, predicted_p_growth, predicted_mask = self._infer_topology_greedily(opt_prob, grammar_guide)
            self._arrange_predicted_tokens(token_stack, exact_token_logit, lhs_mask, predicted_p_growth, predicted_mask)
            self._arrange_predicted_symbols(symbol_stack, lhs_mask, predicted_symbol, predicted_mask)
            self.npda.push_predictions_onto_stack(predicted_symbol, predicted_p_growth, predicted_mask, lhs_mask)

        # compute metrics
        self._compute_err(token_stack, target_tokens)

        exact_tokens, token_mask = token_stack.dump()
        exact_tokens = exact_tokens.squeeze(-1)

        symbols, symbol_mask = symbol_stack.dump()
        symbols = symbols.squeeze(-1)

        output = {
            "source": source_tokens,
            "target": target_tokens,
            "prediction": exact_tokens * token_mask,
            "symbols": symbols * symbol_mask,
            "rhs_symbols": rhs_symbols,
        }

        self.npda.reset_automata()
        return output

    def make_human_readable_output(self, output):
        output['source_surface'] = make_human_readable_text(output['source'], self.vocab, self.src_ns)
        output['target_surface'] = make_human_readable_text(output['target'], self.vocab, self.tgt_ns[1])
        output['prediction_surface'] = make_human_readable_text(output['prediction'], self.vocab, self.tgt_ns[1])
        output['symbol_surface'] = make_human_readable_text(output['symbols'], self.vocab, self.tgt_ns[0])
        output['rhs_symbol_surface'] = make_human_readable_text(output['rhs_symbols'], self.vocab, self.tgt_ns[0])
        return output



