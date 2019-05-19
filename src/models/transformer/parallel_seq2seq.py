from typing import Dict, List, Tuple, Mapping, Optional

import numpy
import torch
import torch.nn
import allennlp.models
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TokenEmbedder
from allennlp.nn import util

from allennlp.training.metrics import BLEU

class ParallelSeq2Seq(allennlp.models.Model):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 source_embedding: TokenEmbedder,
                 target_embedding: TokenEmbedder,
                 target_namespace: str = "target_tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 max_decoding_step: int = 50,
                 use_bleu: bool = True,
                 label_smoothing: Optional[float] = None,
                 output_projection_layer = None,
                 output_is_logic = True,
                 ):
        super(ParallelSeq2Seq, self).__init__(vocab)
        self._encoder = encoder
        self._decoder = decoder
        self._src_embedding = source_embedding
        self._tgt_embedding = target_embedding

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step

        self._target_namespace = target_namespace
        self._label_smoothing = label_smoothing

        if output_projection_layer is None:
            self._output_projection_layer = torch.nn.Linear(decoder.hidden_dim,
                                                            vocab.get_vocab_size(target_namespace))
        else:
            self._output_projection_layer = output_projection_layer
        self._output_is_logic = output_is_logic

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

        if target_tokens is not None and self.training:
            target, target_mask = target_tokens['tokens'], util.get_text_field_mask(target_tokens)

            predictions, logits = self._forward_training(state, target[:, :-1], source_mask, target_mask[:, :-1])
            if self._output_is_logic:
                loss = util.sequence_cross_entropy_with_logits(logits,
                                                               target[:, 1:].contiguous(),
                                                               target_mask[:, 1:].float(),
                                                               label_smoothing=self._label_smoothing)
            else:
                loss = sequence_cross_entropy_with_probs(logits,
                                                         target[:, 1:].contiguous(),
                                                         target_mask[:, 1:].float(),
                                                         label_smoothing=self._label_smoothing)
            if self._bleu:
                self._bleu(predictions, target[:, 1:])

        elif target_tokens is not None: # validation, requires model in eval mode
            target, target_mask = target_tokens['tokens'], util.get_text_field_mask(target_tokens)
            predictions, logits = self._forward_prediction(state, source_mask)
            max_len = min(logits.size()[1], target.size()[1] - 1)
            loss = util.sequence_cross_entropy_with_logits(logits[:, :max_len, :].contiguous(),
                                                           target[:, 1:(max_len + 1)].contiguous(),
                                                           target_mask[:, 1:(max_len + 1)].float().contiguous(),
                                                           label_smoothing=self._label_smoothing)
        else: # testing
            predictions, logits = self._forward_prediction(state, source_mask)
            loss = None

        output = {
            "predictions": predictions,
            "logits": logits,
        }
        if loss is not None:
            output['loss'] = loss

        return output

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert the predicted word ids into discrete tokens"""
        # predictions: (batch, max_length)
        predictions = output_dict["predictions"]
        if not isinstance(predictions, numpy.ndarray):
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
        source_hidden = self._encoder(source_embedding, source_mask)
        return source_hidden

    def _forward_training(self,
                          state: torch.Tensor,
                          target: torch.LongTensor,
                          source_mask: Optional[torch.LongTensor],
                          target_mask: Optional[torch.LongTensor]
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run decoder for training, given target tokens as supervision.
        When training, all timesteps are used and computed universally.
        """
        # target_embedding: (batch, max_target_length, embedding_dim)
        # target_hidden:    (batch, max_target_length, hidden_dim)
        # logits:           (batch, max_target_length, vocab_size)
        # predictions:      (batch, max_target_length)
        target_embedding = self._tgt_embedding(target)
        target_hidden = self._decoder(target_embedding, target_mask, state, source_mask)
        logits = self._output_projection_layer(target_hidden)
        predictions = torch.argmax(logits, dim=-1).detach_()

        return predictions, logits

    def _forward_prediction(self,
                            state: torch.Tensor,
                            source_mask: Optional[torch.LongTensor],
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run decoder step by step for testing or validation, with no gold tokens available.
        """
        batch_size = state.size()[0]
        # batch_start: (batch,)
        batch_start = source_mask.new_ones((batch_size,), dtype=torch.long) * self._start_id

        # step_logits: a list of logits, [(batch, seq_len, vocab_size)]
        logits_by_step = []
        # a list of predicted token ids at each step: [(batch,)]
        predictions_by_step = [batch_start]

        for timestep in range(self._max_decoding_step):
            # step_inputs: (batch, timestep + 1), i.e., at least 1 token at step 0
            # inputs_embedding: (batch, seq_len, embedding_dim)
            # step_hidden:      (batch, seq_len, hidden_dim)
            # step_logit:       (batch, seq_len, vocab_size)
            step_inputs = torch.stack(predictions_by_step, dim=1)
            inputs_embedding = self._tgt_embedding(step_inputs)
            step_hidden = self._decoder(inputs_embedding, None, state, source_mask)
            step_logit = self._output_projection_layer(step_hidden)

            # a list of logits, [(batch, vocab_size)]
            logits_by_step.append(step_logit[:, -1, :])

            # greedy decoding
            # prediction: (batch, seq_len)
            # step_prediction: (batch, )
            prediction = torch.argmax(step_logit, dim=-1)
            step_prediction = prediction[:, -1]
            predictions_by_step.append(step_prediction)

        # predictions: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        predictions = torch.stack(predictions_by_step[1:], dim=1).detach_()
        logits = torch.stack(logits_by_step, dim=1)

        return predictions, logits

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics

def sequence_cross_entropy_with_probs(probs: torch.FloatTensor,
                                      targets: torch.LongTensor,
                                      weights: torch.FloatTensor,
                                      average: str = "batch",
                                      label_smoothing: float = None) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    label_smoothing : ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.

    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
    If ``average is None``, the returned loss is a vector of shape (batch_size,).

    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # shape : (batch * sequence_length, num_classes)
    probs_flat = probs.view(-1, probs.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = (probs_flat + 1e-16).log()
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = probs_flat.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss


