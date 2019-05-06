from typing import Union, Optional, Dict
import torch
import torch.nn

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TokenEmbedder
from allennlp.nn.util import masked_softmax
from allennlp.nn import util as allen_util
from utils.nn import AllenNLPAttentionWrapper, filter_cat, filter_sum
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper
from models.base_s2s.base_seq2seq import BaseSeq2Seq
from training.tree_acc_metric import TreeAccuracy

from models.modules.stacked_encoder import StackedEncoder
from models.modules.stacked_rnn_cell import StackedRNNCell

class UncSeq2Seq(BaseSeq2Seq):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: StackedEncoder,
                 decoder: StackedRNNCell,
                 word_projection: torch.nn.Module,
                 source_embedding: TokenEmbedder,
                 target_embedding: TokenEmbedder,
                 target_namespace: str = "target_tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 max_decoding_step: int = 50,
                 use_bleu: bool = True,
                 label_smoothing: Optional[float] = None,
                 enc_attention: Union[AllenNLPAttentionWrapper, SingleTokenMHAttentionWrapper, None] = None,
                 dec_hist_attn: Union[AllenNLPAttentionWrapper, SingleTokenMHAttentionWrapper, None] = None,
                 scheduled_sampling_ratio: float = 0.,
                 intermediate_dropout: float = .1,
                 concat_attn_to_dec_input: bool = False,
                 model_mode: int = 0,
                 max_pondering: int = 3,
                 uncertainty_sample_num: int = 10,
                 uncertainty_loss_weight: int = 1.,
                 reinforcement_discount: float = 0.,
                 skip_loss: bool = False,
                 inspection: bool = False,
                 ):
        super(UncSeq2Seq, self).__init__(vocab, encoder, decoder, word_projection,
                                         source_embedding, target_embedding, target_namespace,
                                         start_symbol, eos_symbol, max_decoding_step,
                                         use_bleu, label_smoothing,
                                         enc_attention, dec_hist_attn,
                                         scheduled_sampling_ratio, intermediate_dropout,
                                         concat_attn_to_dec_input,
                                         )

        # training step in uncertainty training:
        # 0: pretrain the traditional seq2seq model
        # 1: fix the pretrained s2s model, train the additional module to control structures
        # 2: joint training
        self._model_mode = model_mode
        self._tree_acc = TreeAccuracy(lambda x: self.decode({"predictions": x})["predicted_tokens"])
        self._max_pondering = max_pondering

        self._unc_est_num = uncertainty_sample_num

        self._unc_loss_weight = uncertainty_loss_weight

        # self._reward_discount = reinforcement_discount
        self.skip_loss = skip_loss
        self.inspection = inspection

        out_dim = self._decoder.hidden_dim
        self.unc_fn = torch.nn.Sequential(
            torch.nn.Linear(out_dim, out_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(out_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self,
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """Run the network, and dispatch work to helper functions based on the runtime"""

        # source: (batch, source_length), containing the input word IDs
        # target: (batch, target_length), containing the output IDs

        source, source_mask = source_tokens['tokens'], allen_util.get_text_field_mask(source_tokens)
        state, layer_states = self._encode(source, source_mask)
        init_hidden, _ = self._init_hidden_states(layer_states, source_mask)

        if target_tokens is not None:
            target, target_mask = target_tokens['tokens'], allen_util.get_text_field_mask(target_tokens)

            # predictions: (batch, seq_len)
            # logits: (batch, seq_len, vocab_size)
            predictions, logits, others = self._forward_loop(state, source_mask, init_hidden, target, target_mask)
            loss = self._get_loss(target, target_mask.float(), logits, others)
            self._compute_metric(predictions, target[:, 1:])

        else:
            predictions, logits, others = self._forward_loop(state, source_mask, init_hidden, None, None)
            loss = [-1]

        output = {
            "predictions": predictions,
            "logits": logits,
            "loss": loss,
        }

        if self.inspection:
            _, _, alive_mask_segment = zip(*others)
            all_ponder_step = torch.stack([alive.sum(1) for alive in alive_mask_segment], dim=-1)
            output['n_ponder'] = all_ponder_step.cpu().tolist()

        return output

    def _run_decoder(self, step_target, inputs_embedding, last_hidden, last_output, enc_attn_fn, dec_hist_attn_fn):
        # ponder with the output
        batch = inputs_embedding.size()[0]
        enc_context, dec_hist_context = None, None
        alive = inputs_embedding.new_ones(batch, 1, requires_grad=False)
        zeros = inputs_embedding.new_zeros(batch, 1, requires_grad=False)
        ones = inputs_embedding.new_ones(batch, 1, requires_grad=False)

        # all_action_log_probs, all_action_rewards, all_alive_mask = [(batch, 1)]
        all_action_log_probs, all_action_stock, all_alive_mask = [], [], []
        for pondering_step in range(self._max_pondering):
            pondering_flag = zeros if pondering_step == 0 else ones
            all_alive_mask.append(alive)

            # compute attention context before the output is updated
            enc_context = enc_attn_fn(last_output) if enc_attn_fn else None
            dec_hist_context = dec_hist_attn_fn(last_output) if dec_hist_attn_fn else None

            # step_hidden: some_hidden_var_with_unknown_internals
            # step_output: (batch, hidden_dim)
            cat_context = [
                self._dropout(enc_context) if self._concat_attn and enc_context is not None else None,
                self._dropout(dec_hist_context) if self._concat_attn and dec_hist_context is not None else None,
            ]
            dec_output = self._decoder(inputs_embedding, last_hidden, cat_context + [pondering_flag])
            new_hidden, new_output = dec_output[:2]

            if self._model_mode == 0:
                last_hidden, last_output = new_hidden, new_output
                break

            # compute the probability that we need to continue (1 means we are uncertain, and need to go further)
            # halting_prob, choice: (batch, 1)
            if self._model_mode == 1:   # mode 1 deals only with the uncertainty part
                new_output = new_output.detach()

            halting_prob = self.unc_fn(new_output)
            choice = halting_prob.bernoulli()
            if step_target is not None and not self.skip_loss:
                action_probs = halting_prob * choice + (1 - halting_prob) * (1 - choice)
                all_action_log_probs.append((action_probs + 1e-20).log())
                # compute for the current step the rewards
                step_unc: (batch,)
                step_unc = self._get_step_uncertainty(step_target, inputs_embedding, last_hidden,
                                                      cat_context, pondering_flag)
                step_unc = step_unc.unsqueeze(-1)
                all_action_stock.append(step_unc)

            # if an item within this batch is alive, the new_output will be used next time,
            # otherwise, the last_output will be retained.
            # i.e., last_output = new_output if alive == 1, otherwise last_output = last_output if alive == 0
            last_output = last_output * (1 - alive) + new_output * alive
            last_hidden = self._decoder.merge_hidden_list([last_hidden, new_hidden],
                                                          torch.cat([1 - alive, alive], dim=1))

            # udpate survivors for the next time step
            alive = alive * choice

        step_logit = self._get_step_projection(last_output, enc_context, dec_hist_context)

        # reprocess the reward
        alive_mask = torch.cat(all_alive_mask, dim=-1)
        if self._model_mode > 0 and step_target is not None and not self.skip_loss:
            # only the final correctness matters
            correctness = (step_logit.argmax(dim=-1) == step_target).float().unsqueeze(-1)
            action_log_probs = torch.cat(all_action_log_probs, dim=-1) * alive_mask * correctness
            action_stock = torch.cat(all_action_stock, dim=-1)

            action_rewards = masked_softmax(action_stock, alive_mask)
        else:
            action_rewards, action_log_probs = None, None

        return last_hidden, last_output, step_logit, action_log_probs, action_rewards, alive_mask

    def _get_step_uncertainty(self, step_target, inputs_embedding, last_hidden, cat_context, pondering_flag):
        with torch.no_grad():
            is_orignally_training = self.training
            if not self.training:
                self.train()

            # step_target -> (batch, 1)
            if len(step_target.size()) == 1:
                step_target = step_target.unsqueeze(-1)

            all_pass_prob = []
            for _ in range(self._unc_est_num):
                dec_output = self._decoder(inputs_embedding, last_hidden, cat_context + [pondering_flag])
                _, new_output = dec_output[:2]

                if self._concat_attn:
                    proj_input = filter_cat([new_output] + cat_context, dim=-1)
                else:
                    proj_input = filter_sum([new_output] + cat_context)

                proj_input = self._dropout(proj_input)
                step_logit = self._output_projection(proj_input)
                step_prob = step_logit.softmax(-1)

                # (batch, 1)
                one_pass_prob = step_prob.gather(dim=-1, index=step_target)
                all_pass_prob.append(one_pass_prob)

            # concated_probs: (batch, self._unc_est_num)
            # uncertainty: (batch,)
            concated_probs = torch.cat(all_pass_prob, dim=-1)
            uncertainty = concated_probs.var(dim=1)

            # scaled_unc = torch.stack([uncertainty, torch.full_like(uncertainty, 0.15)], dim=1).max(-1)[0] / 0.15

            if not is_orignally_training:
                self.eval()

            return uncertainty

    def _get_loss(self, target, target_mask, logits, others_by_step):
        if self.skip_loss:
            return 0

        loss_nll = super(UncSeq2Seq, self)._get_loss(target, target_mask, logits, others_by_step)
        if self._model_mode == 0:
            return loss_nll

        action_log_probs_segment, action_rewards_segment, _ = zip(*others_by_step)
        action_log_probs = torch.cat(action_log_probs_segment, dim=-1)
        action_rewards = torch.cat(action_rewards_segment, dim=-1)

        assert action_rewards.size() == action_log_probs.size()

        loss_unc = - action_rewards * action_log_probs
        loss_unc = loss_unc.sum(dim=1).mean(dim=0) # sum for the tokens then average within the batch

        if self._model_mode == 1:
            return loss_unc

        if self._model_mode == 2:
            return loss_unc * self._unc_loss_weight + loss_nll

        return 0

    def _compute_metric(self, predictions, labels):
        super(UncSeq2Seq, self)._compute_metric(predictions, labels)
        if self._tree_acc:
            self._tree_acc(predictions, labels, None)

    def get_metrics(self, reset: bool = False):
        metrics = super(UncSeq2Seq, self).get_metrics(reset)
        if self._tree_acc and not self.training:
            metrics.update(self._tree_acc.get_metric(reset))
        return metrics

