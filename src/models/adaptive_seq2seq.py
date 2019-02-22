from typing import Dict, List, Tuple, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn
import allennlp.models
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TokenEmbedder
from allennlp.nn import util

from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper
from models.adaptive_rnn_cell import AdaptiveRNNCell
from allennlp.training.metrics import BLEU
from utils.nn import AllenNLPAttentionWrapper

class AdaptiveSeq2Seq(allennlp.models.Model):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: torch.nn.Module,
                 decoder: AdaptiveRNNCell,
                 source_embedding: TokenEmbedder,
                 target_embedding: TokenEmbedder,
                 target_namespace: str = "target_tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 max_decoding_step: int = 50,
                 use_bleu: bool = True,
                 label_smoothing: Optional[float] = None,
                 attention: Union[AllenNLPAttentionWrapper, SingleTokenMHAttentionWrapper, None] = None,
                 scheduled_sampling_ratio: float = 0.,
                 act_loss_weight: float = 1.,
                 prediction_dropout: float = .1,
                 embedding_dropout: float = .1,
                 ):
        super(AdaptiveSeq2Seq, self).__init__(vocab)
        self._attention = attention
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._encoder = encoder
        hidden_dim = self._encoder.get_output_dim()
        self._decoder = decoder
        self._src_embedding = source_embedding
        self._tgt_embedding = target_embedding

        self._attn_mapping = torch.nn.Linear(hidden_dim, hidden_dim)

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step

        self._target_namespace = target_namespace
        self._label_smoothing = label_smoothing

        self._act_loss_weight = act_loss_weight

        self._pre_projection_dropout = torch.nn.Dropout(prediction_dropout)
        self._embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self._output_projection_layer = torch.nn.Linear(decoder.hidden_dim,
                                                        vocab.get_vocab_size(target_namespace))

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)
            self._bleu = BLEU(exclude_indices={pad_index, self._eos_id, self._start_id})
        else:
            self._bleu = None


    def forward(self,
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """Run the network, and dispatch work to helper functions based on the runtime"""

        # source: (batch, source_length), containing the input word IDs
        # target: (batch, target_length), containing the output IDs

        source, source_mask = source_tokens['tokens'], util.get_text_field_mask(source_tokens)
        state = self._encode(source, source_mask)

        if target_tokens is not None:
            target, target_mask = target_tokens['tokens'], util.get_text_field_mask(target_tokens)

            # predictions: (batch, seq_len)
            # logits: (batch, seq_len, vocab_size)
            # acc_halting_probs: (batch, seq_len)
            # n_updated: (batch, seq_len)
            predictions, logits, acc_halting_probs, n_updates =\
                self._forward_loop(state, source_mask, target[:, :-1], target_mask[:, :-1])

            loss_pred = util.sequence_cross_entropy_with_logits(logits,
                                                                target[:, 1:].contiguous(),
                                                                target_mask[:, 1:].float(),
                                                                label_smoothing=self._label_smoothing)
            if acc_halting_probs is not None and n_updates is not None:
                # acc_halting_probs will get maximized while n_updates will get minimized
                loss_act = (acc_halting_probs * (- n_updates - 1).float() * target_mask[:, 1:].float()).sum()
            else:
                loss_act = 0

            loss = loss_pred + self._act_loss_weight * loss_act
            if self._bleu:
                self._bleu(predictions, target[:, 1:])

        else:
            predictions, logits, acc_halting_probs, n_updates = self._forward_loop(state, source_mask, None, None)
            loss = [-1]

        output = {
            "predictions": predictions,
            "logits": logits,
            "loss": loss,
        }

        return output

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert the predicted word ids into discrete tokens"""
        # predictions: (batch, max_length)
        predictions = output_dict["predictions"]
        if not isinstance(predictions, np.ndarray):
            predictions = predictions.detach().cpu().numpy()
        all_predicted_tokens = []

        for token_ids in predictions:
            if token_ids.ndim > 1:
                token_ids = token_ids[0]

            token_ids = list(token_ids)
            if self._eos_id in token_ids:
                token_ids = token_ids[:token_ids.index(self._eos_id)]
            tokens = [self.vocab.get_token_from_index(token_id, namespace=self._target_namespace)
                      for token_id in token_ids]
            all_predicted_tokens.append(tokens)
        output_dict['predicted_tokens'] = all_predicted_tokens
        return output_dict

    def _encode(self,
                source: torch.LongTensor,
                source_mask: torch.LongTensor) -> torch.Tensor:
        """
        Do the encoder work: embedding + encoder(which adds positional features and do stacked multi-head attention)

        :param source: (batch, max_input_length), source sequence token ids
        :param source_mask: (batch, max_input_length), source sequence padding mask
        :return: source hidden states output from encoder, which has shape
                 (batch, max_input_length, hidden_dim)
        """

        # source_embedding: (batch, max_input_length, embedding_sz)
        source_embedding = self._src_embedding(source)
        source_embedding = self._embedding_dropout(source_embedding)
        source_hidden = self._encoder(source_embedding, source_mask)
        return source_hidden


    def _forward_loop(self,
                      source_state: torch.Tensor,
                      source_mask: Optional[torch.LongTensor],
                      target: Optional[torch.LongTensor],
                      target_mask: Optional[torch.LongTensor],
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Do the decoding process for training and prediction

        :param source_state: (batch, max_input_length, hidden_dim),
        :param source_mask: (batch, max_input_length)
        :param target: (batch, max_target_length)
        :param target_mask: (batch, max_target_length)
        :return:
        """

        # shape: (batch, max_input_sequence_length)
        batch = source_state.size()[0]

        if target is not None:
            num_decoding_steps = target.size()[1]
        else:
            num_decoding_steps = self._max_decoding_step

        # Initialize target predictions with the start index.
        # batch_start: (batch_size,)
        batch_start = source_mask.new_full((batch,), fill_value=self._start_id)
        step_hidden, step_output = self._decoder.init_hidden_states(source_state, source_mask,
                                                                    self._encoder.is_bidirectional())

        # acc_halting_probs: [(batch,)]
        # updated_num_by_step: [(batch,)]
        # step_logits: [(batch, seq_len, vocab_size)]
        # a list of predicted token ids at each step: [(batch,)]
        acc_halting_probs_by_step = []
        updated_num_by_step = []
        logits_by_step = []
        predictions_by_step = [batch_start]
        for timestep in range(num_decoding_steps):
            if self.training and np.random.rand(1).item() < self._scheduled_sampling_ratio:
                # use self-predicted tokens for scheduled sampling in training with _scheduled_sampling_ratio
                # step_inputs: (batch,)
                step_inputs = predictions_by_step[-1]
            elif not self.training or target is None:
                # no target present, maybe in validation
                # step_inputs: (batch,)
                step_inputs = predictions_by_step[-1]
            else:
                # gold choice
                # step_inputs: (batch,)
                step_inputs = target[:, timestep]

            # inputs_embedding: (batch, embedding_dim)
            inputs_embedding = self._tgt_embedding(step_inputs)
            inputs_embedding = self._embedding_dropout(inputs_embedding)

            context = self._attention(step_output, source_state, source_mask)
            context = self._attn_mapping(context)

            inputs_embedding = inputs_embedding + context

            # step_hidden: some_hidden_var_with_unknown_internals
            # step_output: (batch, hidden_dim)
            # step_acc_halting_probs: (batch, )
            step_hidden, step_output, step_acc_halting_probs, _ = self._decoder(
                inputs_embedding, step_hidden,
                lambda out: self._attn_mapping(self._attention(out, source_state, source_mask))
            )

            if step_acc_halting_probs is not None:
                acc_halting_probs_by_step.append(step_acc_halting_probs)

            # step_logit: (batch, vocab_size)
            step_logit = self._output_projection_layer(self._pre_projection_dropout(step_output + context))

            logits_by_step.append(step_logit)

            # greedy decoding
            # step_prediction: (batch, )
            step_prediction = torch.argmax(step_logit, dim=-1)
            predictions_by_step.append(step_prediction)

        # predictions: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        # acc_halting_probs: (batch, seq_len)
        # n_updated: (batch, seq_len)
        predictions = torch.stack(predictions_by_step[1:], dim=1)
        logits = torch.stack(logits_by_step, dim=1)
        acc_halting_probs = torch.stack(acc_halting_probs_by_step, dim=1) if acc_halting_probs_by_step else None
        n_updated = torch.stack(updated_num_by_step, dim=1) if updated_num_by_step else None

        return predictions, logits, acc_halting_probs, n_updated

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu: # and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics

