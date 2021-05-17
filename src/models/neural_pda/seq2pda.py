from typing import Optional, Tuple, Generic, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.stacked_encoder import StackedEncoder
from allennlp.training.metrics.perplexity import Perplexity, Average
from utils.nn import prepare_input_mask, seq_cross_ent, init_state_for_stacked_rnn
from .npda import NeuralPDA
from utils.seq_collector import SeqCollector
from utils.text_tool import make_human_readable_text
from .batched_stack import TensorBatchStack
import logging
from ..modules.variational_dropout import VariationalDropout

from .tensor_typing_util import *

class Seq2PDA(nn.Module):
    def __init__(self,
                 # modules
                 encoder: StackedEncoder,
                 src_embedding: nn.Embedding,
                 enc_attn_net: nn.Module,
                 enc_attn_mapping: nn.Module,
                 npda: NeuralPDA,

                 # configuration
                 src_ns: str,
                 tgt_ns: List[str],
                 vocab,
                 return_attn_weights: bool = False,
                 ):
        super().__init__()
        self.encoder = encoder
        self.src_embedder = src_embedding
        self.enc_attn_net = enc_attn_net
        self.enc_attn_mapping = enc_attn_mapping
        self.return_attn_weights = return_attn_weights
        self.npda = npda

        self.token_loss = Average()
        self.topo_loss = Average()
        self.attn_loss = Average()
        self.count_metric = 0
        self.err = Average()
        self.tok_pad = 0
        self.src_ns = src_ns
        self.tgt_ns = tgt_ns
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vocab = vocab

        # ------ diagnosis metrics ---------
        # split the
        self.tree_hid_norm = (Average(), Average(), Average())
        self.tree_att_norm = (Average(), Average(), Average())

    def get_metric(self, reset=False, diagnosis=False):
        token_loss = self.token_loss.get_metric(reset)
        topo_loss = self.topo_loss.get_metric(reset)
        attn_loss = self.attn_loss.get_metric(reset)
        count = self.count_metric
        if reset:
            self.count_metric = 0
        err = self.err.get_metric(reset)
        output = {"TokenLoss": token_loss, "TopoLoss": topo_loss, "AttnLoss": attn_loss, "ERR": err, "COUNT": count}

        if diagnosis:
            tree_hid_norm = [m.get_metric(reset) for m in self.tree_hid_norm]
            tree_att_norm = [m.get_metric(reset) for m in self.tree_att_norm]
            output.update(TreeHidNorm=tree_hid_norm, TreeAttNorm=tree_att_norm)
        return output

    def forward(self, *args, **kwargs):
        self._reset_variational_dropout()
        if self.training:
            return self._forward_parallel_training(*args, **kwargs)

        else:
            output = self._forward_parallel_training(*args, **kwargs)
            if 'loss' in output:
                del output['loss']
            return output

    def _forward_parallel_training(self,
                                   source_tokens: LT,  # (batch, src_len)
                                   tree_nodes,  # (batch, n_d), the tree nodes = #lhs = num_derivations
                                   node_parents,  # (batch, n_d),
                                   expansion_frontiers,  # (batch, n_d, max_runtime_stack_size),
                                   derivations,  # (batch, n_d, max_seq), the gold derivations for choice checking
                                   exact_tokens, # (batch, n_d, max_seq)
                                   target_tokens, # (batch, max_tgt_len)
                                   ):
        enc_attn_fn = self._encode_source(source_tokens)
        self.npda.init_automata(source_tokens.size()[0], source_tokens.device, enc_attn_fn)

        # ================
        # opt_prob: (batch, n_d, opt_num)
        # exact_logit: (batch, n_d, max_seq, V)
        # tree_grammar: (batch, n_d, opt_num, 4, max_seq)
        opt_prob, exact_logit, tree_grammar = self.npda(tree_nodes=tree_nodes,
                                                        node_parents=node_parents,
                                                        expansion_frontiers=expansion_frontiers,
                                                        derivations=derivations)

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
        # topo_loss = (nll * tree_mask).sum() / (tree_mask.sum() + 1e-15)
        topo_loss = (nll * tree_mask).sum() / (10 * source_tokens.size()[0])

        # ------------- 2. training the exact token prediction ------------
        _, et_mask = prepare_input_mask(exact_tokens)
        tok_nll = -F.log_softmax(exact_logit, dim=-1).gather(dim=-1, index=exact_tokens.unsqueeze(-1)).squeeze(-1)
        # token_loss = seq_cross_ent(exact_logit, exact_tokens, et_mask, average="token")
        token_loss = (tok_nll * et_mask).sum() / (10 * source_tokens.size()[0])

        # ------------- 3. attention weight penalty ----------------
        # attn_weights: (batch, n_d, src_seq_len)
        if self.return_attn_weights:
            attn_weights = self.enc_attn_net.get_latest_attn_weights()
            attn_loss = F.relu(attn_weights.sum(dim=[1, 2]) - 1).mean()
        else:
            attn_loss = 0

        # ------------- 3. error metric computation -------------
        # *_err: (batch,)
        topo_err = ((opt_prob.argmax(dim=-1) != choice) * tree_mask).sum(dim=-1) + (1 - choice_validity.int())
        token_err = ((exact_logit.argmax(dim=-1) != exact_tokens) * et_mask).sum([1, 2])
        for e1, e2 in zip(topo_err, token_err):
            self.err(1. if e1 + e2 > 0 else 0.)
        self.count_metric += (choice_validity > 0).sum().item()

        # ================
        self._diagnose_tree_state(tree_mask)

        output = {"loss": topo_loss + token_loss + attn_loss}
        self.token_loss(token_loss)
        self.topo_loss(topo_loss)
        self.attn_loss(attn_loss)
        return output

    def _reset_variational_dropout(self):
        for m in self.modules():
            if isinstance(m, VariationalDropout):
                m.reset()

    def _encode_source(self, source_tokens):
        source_tokens, source_mask = prepare_input_mask(source_tokens, padding_val=self.tok_pad)
        source = self.src_embedder(source_tokens)
        source_hidden, _ = self.encoder(source, source_mask)

        def _enc_attn_fn(out):
            context = self.enc_attn_net(out, source_hidden, source_mask)
            context = self.enc_attn_mapping(context)
            if self.return_attn_weights:
                return context, self.enc_attn_net.get_latest_attn_weights()
            else:
                return context

        return _enc_attn_fn

    def _diagnose_tree_state(self, tree_mask):
        tree_hid, tree_att = map(self.npda.diagnosis.get, ('tree_hid', 'tree_hid_att'))
        tree_hid_norm = tree_hid.norm(dim=-1)
        tree_att_norm = tree_att.norm(dim=-1)
        tree_hs = tree_hid_norm.split(50, dim=1)[:3]
        tree_as = tree_att_norm.split(50, dim=1)[:3]
        tree_ms = tree_mask.split(50, dim=1)[:3]
        for i, (h, a, m) in enumerate(zip(tree_hs, tree_as, tree_ms)):
            h = (h * m).sum(dim=-1) / (m.sum(dim=-1) + 1e-13)
            a = (a * m).sum(dim=-1) / (m.sum(dim=-1) + 1e-13)
            m = m.sum(dim=-1)
            for instance_h, instance_a, instance_m in zip(h, a, m):
                if instance_m > 0:
                    self.tree_hid_norm[i](instance_h)
                    self.tree_att_norm[i](instance_a)

    def forward_inference(self,
                          source_tokens: LT,  # (batch, src_len)
                          tree_nodes,  # (batch, n_d), the tree nodes = #lhs = num_derivations
                          node_parents,  # (batch, n_d),
                          expansion_frontiers,  # (batch, n_d, max_runtime_stack_size),
                          derivations,  # (batch, n_d, max_seq), the gold derivations for choice checking
                          exact_tokens,  # (batch, n_d, max_seq)
                          target_tokens,  # (batch, max_tgt_len)
                          ):
        batch_sz, device = source_tokens.size()[0], source_tokens.device
        enc_attn_fn = self._encode_source(source_tokens)
        self.npda.init_automata(batch_sz, device, enc_attn_fn, max_derivation_step=tree_nodes.size()[-1])

        token_stack = TensorBatchStack(batch_sz, 500, item_size=1, dtype=torch.long, device=device)
        symbol_stack = TensorBatchStack(batch_sz, 1000, item_size=1, dtype=torch.long, device=device)
        symbol_list = []
        while self.npda.continue_derivation():
            lhs_idx, lhs_mask, grammar_guide, opt_prob, exact_token_logit = self.npda()

            predicted_symbol, predicted_p_growth, predicted_mask = self._infer_topology_greedily(opt_prob, grammar_guide)
            symbol_list.append(predicted_symbol)
            self._arrange_predicted_tokens(token_stack, exact_token_logit, lhs_mask, predicted_p_growth, predicted_mask)
            self._arrange_predicted_symbols(symbol_stack, lhs_mask, predicted_symbol, predicted_mask)
            self.npda.update_pda_with_predictions(lhs_idx, predicted_symbol, predicted_p_growth, predicted_mask, lhs_mask)

        predictions = torch.stack([F.pad(t, [0, derivations.size()[-1] - t.size()[-1]]) for t in symbol_list], dim=1)
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
            "rhs_symbols": tree_nodes,
        }

        self.npda.reset_automata()
        return output

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

    def make_human_readable_output(self, output):
        output['source_surface'] = make_human_readable_text(output['source'], self.vocab, self.src_ns)
        output['target_surface'] = make_human_readable_text(output['target'], self.vocab, self.tgt_ns[1])
        output['prediction_surface'] = make_human_readable_text(output['prediction'], self.vocab, self.tgt_ns[1])
        output['symbol_surface'] = make_human_readable_text(output['symbols'], self.vocab, self.tgt_ns[0])
        output['rhs_symbol_surface'] = make_human_readable_text(output['rhs_symbols'], self.vocab, self.tgt_ns[0])
        return output



