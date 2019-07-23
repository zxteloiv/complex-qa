from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import re
import torch
import torch.nn
from training.trial_bot import NSVocabulary, PADDING_TOKEN
from allennlp.modules import TokenEmbedder
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper
from models.modules.stacked_rnn_cell import StackedRNNCell
from utils.nn import AllenNLPAttentionWrapper, filter_cat, prepare_input
from allennlp.training.metrics import BLEU
from models.modules.stacked_encoder import StackedEncoder

class Seq2KeywordSeq(torch.nn.Module):
    def __init__(self,
                 vocab: NSVocabulary,
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
                 output_is_logit: bool = True,
                 hidden_states_strategy: str = "avg_lowest"
                 ):
        super(Seq2KeywordSeq, self).__init__()
        self.vocab = vocab
        self._enc_attn = enc_attention
        self._dec_hist_attn = dec_hist_attn

        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._encoder = encoder
        self._decoder = decoder
        self._src_embedding = source_embedding
        self._tgt_embedding = target_embedding

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step

        self._target_namespace = target_namespace
        self._label_smoothing = label_smoothing

        self._output_projection = word_projection

        self._output_is_logit = output_is_logit

        self._dropout = torch.nn.Dropout(intermediate_dropout)

        if use_bleu:
            pad_index = self.vocab.get_token_index(PADDING_TOKEN, target_namespace)
            self._bleu = BLEU(exclude_indices={pad_index, self._eos_id, self._start_id})
        else:
            self._bleu = None

        self._hidden_states_strategy = hidden_states_strategy


    def forward(self,
                source_tokens: torch.LongTensor,
                target_tokens: torch.LongTensor = None,
                src_keyword_tokens: torch.LongTensor = None,
                tgt_keyword_tokens: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """Run the network, and dispatch work to helper functions based on the runtime"""

        # source: (batch, source_length), containing the input word IDs
        # target: (batch, target_length), containing the output IDs

        source, source_mask = prepare_input(source_tokens)
        src_keyword, src_keyword_mask = prepare_input(src_keyword_tokens)
        src_hidden, layered_hidden = self._encode(source, source_mask)
        target, target_mask = prepare_input(target_tokens)
        tgt_keyword, tgt_keyword_mask = prepare_input(tgt_keyword_tokens)

        init_hidden, _ = self._init_hidden_states(layered_hidden, source_mask)

        loss = None
        if target is not None and self.training:
            # predictions: (batch, seq_len)
            # logits: (batch, seq_len, vocab_size)
            predictions, logits, _ = self._forward_loop(src_hidden, target, tgt_keyword,
                                                        source_mask, target_mask, tgt_keyword_mask,
                                                        init_hidden)
            loss = self._get_training_loss(logits, target[:, 1:], target_mask[:, 1:].float(), None)
            self._compute_metric(predictions, target[:, 1:])
        elif target is not None:
            predictions, logits, _ = self._forward_loop(src_hidden, None, tgt_keyword,
                                                        source_mask, None, tgt_keyword_mask,
                                                        init_hidden)
            self._compute_metric(predictions, target[:, 1:])
        else:
            predictions, logits, _ = self._forward_loop(src_hidden, None, tgt_keyword,
                                                        source_mask, None, tgt_keyword_mask,
                                                        init_hidden)

        output = {
            "predictions": predictions,
            "logits": logits,
            "metric": self.get_metrics()
        }
        if loss is not None:
            output['loss'] = loss

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
                source_mask: torch.LongTensor):
        # source: (batch, max_input_length), source sequence token ids
        # source_mask: (batch, max_input_length), source sequence padding mask
        # source_embedding: (batch, max_input_length, embedding_sz)
        source_embedding = self._src_embedding(source)
        source_embedding = self._dropout(source_embedding)
        source_hidden, layered_hidden = self._encoder(source_embedding, source_mask)
        return source_hidden, layered_hidden

    def _forward_loop(self,
                      src_hidden: torch.Tensor,
                      tgt: Optional[torch.LongTensor],
                      tgt_kwd: torch.LongTensor,
                      src_mask: torch.LongTensor,
                      tgt_mask: Optional[torch.LongTensor],
                      tgt_kwd_mask: torch.LongTensor,
                      init_hidden: torch.Tensor,
                      ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Do the decoding process for training and prediction

        :param src_hidden:  (batch, src_len, hidden)
        :param kwd_hidden:  (batch, kwd_len, hidden)
        :param tgt:         (batch, tgt_len)
        :param tgt_kwd:     (batch, constraint_len)
        :param src_mask:    (batch, src_len)
        :param kwd_mask:    (batch, kwd_len)
        :param tgt_mask:    (batch, tgt_len)
        :param tgt_kwd_mask:(batch, constraint_len)
        :return:
        """

        # shape: (batch, max_input_sequence_length)
        batch = src_hidden.size()[0]

        if tgt is not None:
            num_decoding_steps = tgt.size()[1] - 1
        else:
            num_decoding_steps = self._max_decoding_step

        # Initialize target predictions with the start index.
        # batch_start: (batch_size,)
        batch_start = src_mask.new_full((batch,), self._start_id)
        step_hidden, step_output = init_hidden, self._decoder.get_output_state(init_hidden)

        if self._enc_attn is not None:
            enc_attn_fn = lambda out: self._enc_attn(out, src_hidden, src_mask)
        else:
            enc_attn_fn = None

        # acc_halting_probs: [(batch,)]
        # updated_num_by_step: [(batch,)]
        # step_logits: [(batch, seq_len, vocab_size)]
        # a list of predicted token ids at each step: [(batch,)]
        logits_by_step = []
        output_by_step = []
        others_by_step = []
        predictions_by_step = [batch_start]
        cond_cursor = src_mask.new_zeros((batch,))
        for timestep in range(num_decoding_steps):
            if self.training and np.random.rand(1).item() < self._scheduled_sampling_ratio:
                # use self-predicted tokens for scheduled sampling in training with _scheduled_sampling_ratio
                # step_inputs: (batch,)
                step_inputs = predictions_by_step[-1]
            elif not self.training or tgt is None:
                # no target present, maybe in validation
                # step_inputs: (batch,)
                step_inputs = predictions_by_step[-1]
            else:
                # gold choice
                # step_inputs: (batch,)
                step_inputs = tgt[:, timestep]

            # inputs_embedding: (batch, embedding_dim)
            inputs_embedding = self._tgt_embedding(step_inputs)
            inputs_embedding = self._dropout(inputs_embedding)

            # tgt_kwd: (batch, keywords_count)
            # cond_cursor: (batch,)
            # step_cond_inp: (batch,) <- (batch, 1)
            # cond_embedding: (batch, embedding_dim)
            cond_cursor = find_next_keyword(step_inputs, cond_cursor, tgt_kwd)
            step_cond_inp = tgt_kwd.gather(dim=-1, index=cond_cursor.unsqueeze(1)).squeeze(1)
            cond_embedding = self._tgt_embedding(step_cond_inp)
            cond_embedding = self._dropout(cond_embedding)

            decoder_inp = torch.cat([inputs_embedding, cond_embedding], dim=-1)

            if self._dec_hist_attn is None:
                dec_hist_attn_fn = None

            elif len(output_by_step) > 0:
                dec_hist = torch.stack(output_by_step, dim=1)
                dec_hist_mask = tgt_mask[:, :timestep] if tgt_mask is not None else None
                dec_hist_attn_fn = lambda out: self._dec_hist_attn(out, dec_hist, dec_hist_mask)

            else:
                dec_hist_attn_fn = lambda out: torch.zeros_like(out)

            dec_out = self._run_decoder(decoder_inp, step_hidden, step_output, enc_attn_fn, dec_hist_attn_fn)
            step_hidden, step_output, step_logit = dec_out[:3]
            if len(dec_out) > 3:
                others_by_step.append(dec_out[3:])

            output_by_step.append(step_output)
            logits_by_step.append(step_logit)

            # greedy decoding
            # step_prediction: (batch, )
            step_prediction = torch.argmax(step_logit, dim=-1)
            predictions_by_step.append(step_prediction)

        # predictions: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        predictions = torch.stack(predictions_by_step[1:], dim=1)
        logits = torch.stack(logits_by_step, dim=1)

        return predictions, logits, others_by_step

    def _run_decoder(self, inputs_embedding, step_hidden, step_output, enc_attn_fn, dec_hist_attn_fn):
        # compute attention context before the output is updated
        enc_context = enc_attn_fn(step_output) if enc_attn_fn else None
        dec_hist_context = dec_hist_attn_fn(step_output) if dec_hist_attn_fn else None

        # step_hidden: some_hidden_var_with_unknown_internals
        # step_output: (batch, hidden_dim)
        cat_context = []
        if enc_context is not None:
            cat_context.append(self._dropout(enc_context))
        if dec_hist_context is not None:
            cat_context.append(self._dropout(dec_hist_context))
        dec_output = self._decoder(inputs_embedding, step_hidden, cat_context)
        step_hidden, step_output = dec_output[:2]

        step_logit = self._get_step_projection(step_output, enc_context, dec_hist_context)

        return step_hidden, step_output, step_logit

    def _get_step_projection(self, *inputs):
        # step_logit: (batch, vocab_size)
        proj_input = filter_cat(inputs, dim=-1)
        proj_input = self._dropout(proj_input)
        step_logit = self._output_projection(proj_input)
        return step_logit

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu: # and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics

    def _get_training_loss(self, logits, target, target_mask, other):
        # conventional cross entropy loss
        # num_non_empty_sequences: (1,)
        num_non_empty_sequences = ((target_mask.sum(1) > 0).float().sum() + 1e-20)

        # log_prob: (batch, length, vocab)
        if self._output_is_logit:
            log_probs = torch.log_softmax(logits, -1)
        else:
            log_probs = torch.log(logits + 1e-20)

        # pos_log_prob: (batch, length, 1) -> (batch, length)
        pos_log_prob = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        # negative alpha indicates traditional NLL loss
        # sum the loss of all tokens of each sentence, and rescales the loss for each sentence,
        # sum and divide it by the sentence amount.
        # per_batch_loss: (batch, )
        # num_non_empty_sequences: (1,)
        # loss_lm_margin: (1,)
        per_batch_loss = - (pos_log_prob * target_mask.float()).sum(1) / (target_mask.sum(1).float() + 1e-20)
        loss = per_batch_loss.sum() / num_non_empty_sequences
        return loss

    def _init_hidden_states(self, layer_state, source_mask: torch.LongTensor):
        """
        Initialize the hidden states for decoder given encoders
        :param layer_state: [(batch, src_len, hidden_dim)]
        :param source_mask: (batch, src_len)
        :return:
        """
        # available strategies:
        # [avg|pooling|last|zero]_[lowest|all|parallel]
        # which means
        # 1) to use some aggregation heuristics for the encoder, and
        # 2) apply to the decoder initial hidden states
        m = re.match(r"(avg|max|forward_last|zero)_(lowest|all|parallel)", self._hidden_states_strategy)
        if not m:
            raise ValueError(f"specified strategy '{str(self._hidden_states_strategy)}' not supported")

        agg_stg, assign_stg = m.group(1), m.group(2)
        batch, _, hidden_dim = layer_state[0].size()
        source_mask_expand = source_mask.unsqueeze(-1).float() # (batch, seq_len, hidden)
        if agg_stg == "avg":
            src_agg = [
                (l * source_mask_expand).sum(1) / (source_mask_expand.sum(1) + 1e-30)
                for l in layer_state
            ]

        elif agg_stg == "max":
            src_agg = [
                ((source_mask_expand + 1e-45).log() + l).max(1)
                for l in layer_state
            ]
        elif agg_stg == "zero":
            src_agg = [source_mask.new_zeros((batch, hidden_dim), dtype=torch.float32) for _ in layer_state]
        else: # forward_last
            # last_word_indices: (batch,)
            # expanded_indices: (batch, 1, hidden_dim)
            # forward_by_layer: [(batch, hidden_dim)]
            last_word_indices = source_mask.sum(1).long() - 1
            expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch, 1, hidden_dim)
            forward_by_layer = [state.gather(1, expanded_indices).squeeze(1) for state in layer_state]
            if self._encoder.is_bidirectional():
                hidden_dim = hidden_dim // 2
                src_agg = [state[:, :hidden_dim] for state in forward_by_layer]
            else:
                src_agg = forward_by_layer

        if assign_stg == "lowest": # use the top layer aggregated state for the decoder bottom, zero for others
            decoder_hidden = [src_agg[-1]] + [source_mask.new_zeros((batch, hidden_dim), dtype=torch.float32)
                                              for _ in range(self._decoder.get_layer_num() - 1)]
        elif assign_stg == "all": # use the same top layer state for all decoder layers
            decoder_hidden = [src_agg[-1] for _ in range(self._decoder.get_layer_num())]
        else: # parallel, each encoder is used for the appropriate decoder layer
            decoder_hidden = src_agg

        return self._decoder.init_hidden_states(decoder_hidden)

    def _compute_metric(self, predictions, labels):
        if self._bleu:
            self._bleu(predictions, labels)


def find_next_keyword(last_pred: torch.LongTensor,
                      last_kwd_index: torch.LongTensor,
                      kwd: torch.LongTensor) -> torch.LongTensor:
    """
    Find the keyword to be concatenated at the next timestep,
    given the last predictions and last keywords indices.
    The given keywords for targets are ordered already.
    :param last_pred: (batch,)
    :param last_kwd_index: (batch,)
    :param kwd: (batch, constraint_len)
    :return: next_kwd_index: (batch,)
    """
    # last_kwd: (batch,)
    last_kwd = kwd.gather(dim=1, index=last_kwd_index.unsqueeze(dim=1)).squeeze(dim=1)

    # If the keyword has been successfully predicted at last time, or it has reaches the end,
    # the cursor will keep pointing to the last token in the keyword sequence,
    # which should be some unused special token.
    # pred_success: (batch,)
    # next_kwd_idx: (batch,)
    pred_success = (last_kwd == last_pred).long()
    next_kwd_idx = last_kwd_index + pred_success

    max_kwd_idx = kwd.size()[1]
    # keep the index not exceeding the max keywords
    next_kwd_idx = next_kwd_idx.where(next_kwd_idx < max_kwd_idx, torch.full_like(next_kwd_idx, max_kwd_idx - 1))
    return next_kwd_idx

