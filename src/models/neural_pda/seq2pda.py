import allennlp.nn.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_s2s.encoder_stacker import EncoderStacker
from allennlp.training.metrics.perplexity import Average
from utils.nn import prepare_input_mask, get_final_encoder_states
from .npda import NeuralPDA
from utils.text_tool import make_human_readable_text
from models.modules.batched_stack import TensorBatchStack
import logging
from ..modules.variational_dropout import VariationalDropout
from trialbot.data import NSVocabulary

from .tensor_typing_util import *


class Seq2PDA(nn.Module):
    def __init__(self,
                 # modules
                 encoder: EncoderStacker,
                 src_embedding: nn.Embedding,
                 enc_attn_net: nn.Module,
                 enc_attn_mapping: nn.Module,
                 npda: NeuralPDA,

                 # configuration
                 src_ns: str,
                 tgt_ns: str,
                 vocab,
                 repr_loss_lambda: float = .2,
                 ):
        super().__init__()
        self.encoder = encoder
        self.src_embedder = src_embedding
        self.enc_attn_net = enc_attn_net
        self.enc_attn_mapping = enc_attn_mapping
        self.npda = npda

        self.topo_loss = Average()
        self.repr_loss = Average()
        self.count_metric = 0
        self.err = Average()
        self.tok_pad = 0
        self.src_ns = src_ns
        self.tgt_ns = tgt_ns
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vocab: NSVocabulary = vocab

        self.loss_lambda = repr_loss_lambda

    def get_metric(self, reset=False):
        topo_loss = self.topo_loss.get_metric(reset)
        count = self.count_metric
        err = self.err.get_metric(reset)

        if self.loss_lambda > 0:
            repr_loss = self.repr_loss.get_metric(reset)
            output = {"Loss": topo_loss, "ReprLoss": repr_loss, "ERR": err, "COUNT": count}
        else:
            output = {"Loss": topo_loss, "ERR": err, "COUNT": count}

        if reset:
            self.count_metric = 0
        return output

    def _reset_variational_dropout(self):
        for m in self.modules():
            if isinstance(m, VariationalDropout):
                m.reset()

    def forward(self, model_function: str = "parallel", *args, **kwargs):
        self._reset_variational_dropout()

        if model_function == "parallel":
            return self._forward_parallel_training(*args, **kwargs)
        elif model_function == 'generation':
            return self.forward_inference(*args, **kwargs)
        elif model_function == "validation":
            return self._forward_validation(*args, **kwargs)
        else:
            raise ValueError(f"Unknown model function {model_function} is specified.")

    def _forward_parallel_training(self,
                                   source_tokens: LT,  # (batch, src_len)
                                   tree_nodes,  # (batch, n_d), the tree nodes = #lhs = num_derivations
                                   node_parents,  # (batch, n_d),
                                   expansion_frontiers,  # (batch, n_d, max_runtime_stack_size),
                                   derivations,  # (batch, n_d, max_seq), the gold derivations for choice checking
                                   *args,
                                   **kwargs,
                                   ):
        # enc_attn_fn: (*, hid) -> (*, enc_out_dim)
        # enc_last_repr: (batch, enc_out_dim)
        enc_attn_fn, enc_last_repr = self._encode_source(source_tokens)
        self.npda.init_automata(source_tokens.size()[0], source_tokens.device, enc_attn_fn)

        # ================
        # opt_prob: (batch, n_d, opt_num)
        # opt_repr: (batch, n_d, opt_num, hid)
        # tree_grammar: (batch, n_d, opt_num, 4, max_seq)
        opt_prob, opt_repr = self.npda(model_function="parallel",
                                       tree_nodes=tree_nodes,
                                       node_parents=node_parents,
                                       expansion_frontiers=expansion_frontiers)
        tree_grammar = self.npda.get_grammar(tree_nodes)

        # --------------- 1. training the topological choice --------------
        # valid_symbols, symbol_mask: (batch, n_d, opt_num, max_seq)
        # choice: (batch, n_d)
        # choice_validity: (batch,)
        valid_symbols, _, _, symbol_mask = self.npda.decouple_grammar_guide(tree_grammar)
        choice, choice_validity = self.npda.find_choice_by_symbols(derivations, valid_symbols, symbol_mask)

        # tree_mask, nll: (batch, n_d)
        tree_mask = (tree_nodes != self.tok_pad).long() * choice_validity.unsqueeze(-1).float()
        nll = -(opt_prob + 1e-15).log().gather(dim=-1, index=choice.unsqueeze(-1)).squeeze(-1)
        # use a sensitive loss such that the grad won't get discounted by the factor of 0-loss items
        topo_loss = (nll * tree_mask).sum() / (tree_mask.sum() + 1e-15)
        # topo_loss = (nll * tree_mask).sum() / (10 * source_tokens.size()[0])
        # topo_loss = ((nll * tree_mask).sum(-1) / (tree_mask.sum(-1) + 1e-15)).mean()

        # --------------- 2. the representation regularization --------------
        repr_loss = 0
        if self.loss_lambda > 0:
            batch_sz, node_num = choice.size()
            device = choice.device
            batch_dim_indices = torch.arange(batch_sz, device=device).reshape(-1, 1, 1)
            node_dim_indices = torch.arange(node_num, device=device).reshape(1, -1, 1)
            # chose_repr: (batch, n_d, hid)
            # tree_repr: (batch, hid)
            chosen_repr = opt_repr[batch_dim_indices, node_dim_indices, choice.unsqueeze(-1)].squeeze(-2)
            tree_repr = (tree_mask.unsqueeze(-1) * chosen_repr).sum(dim=1)

            # MSE loss as a regularization for the encoder output and rule repr
            repr_loss = ((self.enc_attn_mapping(enc_last_repr) - tree_repr) ** 2).sum(-1).sqrt().mean()

        # ------------- 3. error metric computation -------------
        # *_err: (batch,)
        topo_pos_err = ((opt_prob.argmax(dim=-1) != choice) * tree_mask)
        topo_err = topo_pos_err.sum(dim=-1) + (1 - choice_validity.int())
        for e in topo_err:
            self.err(1. if e > 0 else 0.)
        self.count_metric += (choice_validity > 0).sum().item()

        # ================
        loss = topo_loss + self.loss_lambda * repr_loss
        output = {"loss": loss}
        self.topo_loss(topo_loss)
        self.repr_loss(repr_loss)
        self.npda.reset_automata()
        return output

    def _encode_source(self, source_tokens):
        source_tokens, source_mask = prepare_input_mask(source_tokens, padding_val=self.tok_pad)
        source = self.src_embedder(source_tokens)
        source_hidden = self.encoder(source, source_mask)
        last_repr = get_final_encoder_states(source_hidden, source_mask, self.encoder.is_bidirectional())

        def _enc_attn_fn(out):
            context = self.enc_attn_net(out, source_hidden, source_mask)
            context = self.enc_attn_mapping(context)
            return context

        return _enc_attn_fn, last_repr

    def _forward_validation(self,
                            source_tokens: LT,  # (batch, src_len)
                            tree_nodes,  # (batch, n_d), the tree nodes = #lhs = num_derivations
                            node_parents,  # (batch, n_d),
                            expansion_frontiers,  # (batch, n_d, max_runtime_stack_size),
                            derivations,  # (batch, n_d, max_seq), the gold derivations for choice checking
                            *args,
                            **kwargs,
                            ):
        enc_attn_fn, enc_last_repr = self._encode_source(source_tokens)
        self.npda.init_automata(source_tokens.size()[0], source_tokens.device, enc_attn_fn)
        # rule_mask: (batch, n_d)
        # rule_logit: (batch, n_d), the score dimension 1 is squeezed by default in the rule scorer
        rule_logit, rule_mask = self.npda(model_function="validation",
                                          tree_nodes=tree_nodes,
                                          node_parents=node_parents,
                                          expansion_frontiers=expansion_frontiers,
                                          derivations=derivations)

        # rule_reward: (batch,)
        rule_reward = allennlp.nn.util.masked_mean(rule_logit, rule_mask, dim=-1)

        self.npda.reset_automata()
        output = {
            "reward": rule_reward
        }
        return output

    def forward_inference(self,
                          source_tokens: LT,  # (batch, src_len)
                          tree_nodes,  # (batch, n_d), the tree nodes = #lhs = num_derivations
                          node_parents,  # (batch, n_d),
                          expansion_frontiers,  # (batch, n_d, max_runtime_stack_size),
                          derivations,  # (batch, n_d, max_seq), the gold derivations for choice checking
                          target_tokens,  # (batch, max_tgt_len)
                          ):
        batch_sz, device = source_tokens.size()[0], source_tokens.device
        enc_attn_fn, enc_last_repr = self._encode_source(source_tokens)
        _, max_derivation_num, max_rule_size = derivations.size()
        self.npda.init_automata(batch_sz, device, enc_attn_fn, max_derivation_step=max_derivation_num)

        symbol_stack = TensorBatchStack(batch_sz, 1000, item_size=1, dtype=torch.long, device=device)
        symbol_list = []
        while self.npda.continue_derivation():
            lhs_idx, lhs_mask, grammar_guide, opt_prob, exact_token_logit = self.npda(model_function="generation")

            predicted_symbol, predicted_p_growth, predicted_mask = self._infer_topology_greedily(opt_prob, grammar_guide)
            symbol_list.append(predicted_symbol)
            self._arrange_predicted_symbols(symbol_stack, lhs_mask, predicted_symbol, predicted_mask)
            self.npda.update_pda_with_predictions(lhs_idx, predicted_symbol, predicted_p_growth, predicted_mask, lhs_mask)

        predictions = torch.stack([F.pad(t, [0, max_rule_size - t.size()[-1]]) for t in symbol_list], dim=1)
        # compute metrics
        # self._compute_err(token_stack, target_tokens)

        symbols, symbol_mask = symbol_stack.dump()
        symbols = symbols.squeeze(-1)

        output = {
            "source": source_tokens,
            "target": target_tokens,
            # "prediction": exact_tokens * token_mask,
            "symbols": symbols * symbol_mask,
            "rhs_symbols": tree_nodes,
        }

        self.npda.reset_automata()
        return output

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

    def make_human_readable_output(self, output):
        output['source_surface'] = make_human_readable_text(output['source'], self.vocab, self.src_ns)
        # output['target_surface'] = make_human_readable_text(output['target'], self.vocab, self.tgt_ns[1])
        # output['prediction_surface'] = make_human_readable_text(output['prediction'], self.vocab, self.tgt_ns[1])
        # output['symbol_surface'] = make_human_readable_text(output['symbols'], self.vocab, self.tgt_ns[0])
        # output['rhs_symbol_surface'] = make_human_readable_text(output['rhs_symbols'], self.vocab, self.tgt_ns[0])
        return output



