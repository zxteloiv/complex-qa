from typing import List, Optional, Tuple, Dict
import torch.nn
import random
import numpy

from allennlp.modules import TokenEmbedder
from allennlp.training.metrics import BLEU
from utils.nn import add_positional_features, prepare_input_mask

from trialbot.data.ns_vocabulary import NSVocabulary, PADDING_TOKEN
from models.transformer.multi_head_attention import MaskedMultiHeadSelfAttention, MultiHeadAttention

class DecoderLayer(torch.nn.Module):
    def __init__(self,
                 input_dim: int,  # input embedding dimension
                 hidden_dim: int = None, # output dimension of the decoder layer
                 num_heads: int = 8,
                 feedforward_hidden_dim: int = None,
                 feedforward_dropout: float = 0.1,
                 attention_dim: int = None,
                 value_dim: int = None,
                 residual_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_positional_embedding: bool = True,
                 use_src_keyword_attention: bool = True,
                 use_shared_keyword_attention: bool = False,
                 ):
        super(DecoderLayer, self).__init__()

        hidden_dim = hidden_dim or input_dim
        self.hidden_dim = hidden_dim
        attention_dim = attention_dim or (hidden_dim // num_heads)
        value_dim = value_dim or (hidden_dim // num_heads)
        feedforward_hidden_dim = feedforward_hidden_dim or hidden_dim

        # output dim is assumed to be equal to input_dim, which is divisible by num_heads
        self._mask_attn = MaskedMultiHeadSelfAttention(
            num_heads, input_dim, attention_dim * num_heads, value_dim * num_heads, attention_dropout
        )

        self._mask_attn_norm = torch.nn.LayerNorm(hidden_dim)
        self._src_attn = MultiHeadAttention(
            num_heads, hidden_dim, hidden_dim, attention_dim * num_heads, value_dim * num_heads,
            attention_dropout=attention_dropout
        )

        self._key_attn = (use_shared_keyword_attention and self._src_attn) or MultiHeadAttention(
            num_heads, hidden_dim, hidden_dim, attention_dim * num_heads, value_dim * num_heads,
            attention_dropout=attention_dropout
        )

        feedforward = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 if use_src_keyword_attention else hidden_dim, feedforward_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(feedforward_dropout),
            torch.nn.Linear(feedforward_hidden_dim, hidden_dim),
        )
        self._feedforward = feedforward
        self._feedforward_norm = torch.nn.LayerNorm(hidden_dim)

        self._dropout = torch.nn.Dropout(residual_dropout)
        self._use_pos_emb = use_positional_embedding
        self._use_src_keyword_attn = use_src_keyword_attention

    def forward(self, src, src_mask, src_kwd, src_kwd_mask, inp, inp_mask):
        """
        Run a layer

        :param src: source sequence, (batch, source_len, hidden)
        :param src_mask: source sequence padding mask, (batch, source_len)
        :param src_kwd: source keyword sequence, (batch, source_keyword_len, hidden)
        :param src_kwd_mask: source keyword sequence padding mask, (batch, source_keyword_len)
        :param inp: input to the decoder layer, (batch, target_len, hidden)
        :param inp_mask: mask of the input to the decoder layer
        :return:
        """
        output = add_positional_features(inp) if self._use_pos_emb else inp

        masked_attn_out, _ = self._mask_attn(output, inp_mask)
        masked_attn_out = self._dropout(masked_attn_out)
        masked_attn_out = self._mask_attn_norm(masked_attn_out + output)

        src_attn_out, _ = self._src_attn(masked_attn_out, src, src_mask)
        key_attn_out, _ = self._key_attn(masked_attn_out, src_kwd, src_kwd_mask)
        attn_out = torch.cat(
            list(map(
                lambda x: self._dropout(x) + masked_attn_out,
                (src_attn_out, key_attn_out)
            )),
            dim=-1
        )
        feedforward_out = self._feedforward(attn_out)
        feedforward_out = self._dropout(feedforward_out)
        feedforward_out = self._feedforward_norm(feedforward_out)

        return feedforward_out

class Decoder(torch.nn.Module):
    def __init__(self, layers: List[DecoderLayer]):
        super(Decoder, self).__init__()
        self._layers = torch.nn.ModuleList(layers)

    def forward(self, src, src_mask, src_kwd, src_kwd_mask, inp, inp_mask):
        for l in self._layers:
            inp = l(src, src_mask, src_kwd, src_kwd_mask, inp, inp_mask)
        return inp

    @property
    def hidden_dim(self):
        return self._layers[-1].hidden_dim

class KeywordConditionedTransformer(torch.nn.Module):
    def __init__(self,
                 vocab: NSVocabulary,
                 source_encoder: torch.nn.Module,
                 source_keyword_encoder: torch.nn.Module,
                 decoder: Decoder,
                 source_embedding: TokenEmbedder,
                 target_embedding: TokenEmbedder,
                 target_namespace: str = "target_tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 max_decoding_step: int = 50,
                 use_bleu: bool = True,
                 label_smoothing: Optional[float] = None,
                 output_projection_layer = None,
                 output_is_logit = False,
                 beam_size: int = 1,
                 diversity_factor: float = 0.,
                 accumulation_factor: float = 1.,
                 use_cross_entropy: bool = True,
                 margin: float = 1.,
                 ):
        super(KeywordConditionedTransformer, self).__init__()
        self.vocab = vocab
        self._src_enc = source_encoder
        self._key_enc = source_keyword_encoder
        self._decoder = decoder
        self._src_embedding = source_embedding
        self._tgt_embedding = target_embedding

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step

        self._target_namespace = target_namespace
        self._label_smoothing = label_smoothing
        self._vocab_size = vocab.get_vocab_size(target_namespace)

        self._decoder_inp_mapping = torch.nn.Linear(decoder.hidden_dim * 2, decoder.hidden_dim)

        self._use_cross_ent = use_cross_entropy
        if output_projection_layer is None:
            self._output_projection_layer = torch.nn.Linear(decoder.hidden_dim, self._vocab_size)
        else:
            self._output_projection_layer = output_projection_layer

        if use_bleu:
            pad_index = self.vocab.get_token_index(PADDING_TOKEN, self._target_namespace)
            self._bleu = BLEU(exclude_indices={pad_index, self._eos_id, self._start_id})
        else:
            self._bleu = None

        self._beam_size = beam_size
        self._diversity_factor = diversity_factor
        self._acc_factor = accumulation_factor

        self._margin = margin
        self._output_is_logit = output_is_logit

    def forward(self,
                source_tokens: torch.LongTensor,
                target_tokens: torch.LongTensor = None,
                src_keyword_tokens: torch.LongTensor = None,
                tgt_keyword_tokens: torch.LongTensor = None,
                ):
        """
        Run the entire model in every circumstances

        :param source_tokens: (batch, source_len)
        :param target_tokens: (batch, target_len)
        :param src_keyword_tokens: (batch, source_key_len) orderded
        :param tgt_keyword_tokens: (batch, target_key_len) orderded
        :return:
        """
        source, source_mask = prepare_input_mask(source_tokens)
        src_keyword, src_keyword_mask = prepare_input_mask(src_keyword_tokens)
        src_hidden, kwd_hidden = self._encode(source, source_mask, src_keyword, src_keyword_mask)

        target, target_mask = prepare_input_mask(target_tokens)
        tgt_keyword, tgt_keyword_mask = prepare_input_mask(tgt_keyword_tokens)

        loss = None
        if target is not None and self.training:    # training mode
            pred, logits = self._forward_training(src_hidden, kwd_hidden, target[:, :-1], tgt_keyword[:, :-1],
                                                  source_mask, src_keyword_mask,
                                                  target_mask[:, :-1], tgt_keyword_mask,)
            self._compute_metric(pred, target[:, 1:])
            loss = self._get_training_loss(logits, target[:, 1:], target_mask[:, 1:])
        elif target is not None:    # evaluation mode
            pred, logits = self._forward_greedy_search(src_hidden, source_mask, kwd_hidden, src_keyword_mask,
                                                       tgt_keyword[:, :-1], tgt_keyword_mask[:, :-1])
            self._compute_metric(pred, target[:, 1:])
            loss = self._get_training_loss(logits, target[:, 1:], target_mask[:, 1:])
        else:   # testing
            pred, logits = self._forward_prediction(src_hidden, source_mask, kwd_hidden, src_keyword_mask,
                                                    tgt_keyword[:, :-1], tgt_keyword_mask[:, :-1])

        output = {
            "predictions": pred,
            "logits": logits,
        }
        if loss is not None:
            output["loss"] = loss
        return output

    def _compute_metric(self, predictions, target):
        if self._bleu:
            self._bleu(predictions, target[:, 1:])

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics

    def _encode(self, src, src_mask, kwd, kwd_mask):
        src_emb = self._src_embedding(src)
        kwd_emb = self._src_embedding(kwd)
        src_hidden = self._src_enc(src_emb, src_mask)
        kwd_hidden = self._key_enc(kwd_emb, kwd_mask)
        return src_hidden, kwd_hidden

    def _forward_training(self,
                          src_hidden: torch.Tensor,
                          kwd_hidden: torch.Tensor,
                          tgt: torch.LongTensor,
                          tgt_kwd: torch.LongTensor,
                          src_mask: torch.LongTensor,
                          kwd_mask: torch.LongTensor,
                          tgt_mask: torch.LongTensor,
                          tgt_kwd_mask: torch.LongTensor,
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training with given target.
        :param src_hidden:  (batch, src_len, hidden)
        :param kwd_hidden:  (batch, kwd_len, hidden)
        :param tgt:         (batch, tgt_len)
        :param tgt_kwd:     (batch, constraint_len)
        :param src_mask:    (batch, src_len)
        :param kwd_mask:    (batch, kwd_len)
        :param tgt_mask:    (batch, tgt_len)
        :param tgt_kwd_mask:(batch, constraint_len)
        :return: A tuple contains predictions and logits.
                prediction: (batch, tgt_len)
                logits:     (batch, tgt_len, vocab)
        """
        # tgt_cond: (batch, tgt_len), keyword sequence same shape as the training target
        # tgt_cond_emb: (batch, tgt_len, hidden)
        tgt_cond = self._build_training_target(tgt, tgt_kwd)
        tgt_cond_emb = self._tgt_embedding(tgt_cond)

        # tgt_emb: (batch, tgt_len, hidden)
        tgt_emb = self._tgt_embedding(tgt)

        # decoder_inp: (batch, tgt_len, hidden * 2) -> (batch, tgt_len, hidden)
        # target_hidden: (batch, tgt_len, hidden)
        # logits: (batch, tgt_len, vocab)
        # predictions: (batch, tgt_len)
        decoder_inp = self._decoder_inp_mapping(torch.cat([tgt_emb, tgt_cond_emb], dim=-1))
        target_hidden = self._decoder(src_hidden, src_mask, kwd_hidden, kwd_mask, decoder_inp, tgt_mask)
        logits = self._output_projection_layer(target_hidden)
        predictions = torch.argmax(logits, dim=-1).detach_()

        return predictions, logits

    def _get_training_loss(self, logits, target, target_mask):
        # num_non_empty_sequences: (1,)
        num_non_empty_sequences = ((target_mask.sum(1) > 0).float().sum() + 1e-20)

        # log_prob: (batch, length, vocab)
        if self._output_is_logit:
            log_probs = torch.log_softmax(logits, -1)
        else:
            log_probs = torch.log(logits + 1e-20)

        # ==================== loss 1: LM as the regularizer ==================
        # construct negative samples
        batch_size = target.size()[0]
        shift_range = random.randint(1, batch_size // 2)
        neg_target = torch.cat((target[-shift_range:], target[:-shift_range]))
        neg_target_mask = torch.cat((target_mask[-shift_range:], target_mask[:-shift_range]))

        # pos_log_prob: (batch, length, 1) -> (batch, length)
        pos_log_prob = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        if self._use_cross_ent:
            # negative alpha indicates traditional NLL loss
            # sum the loss of all tokens of each sentence, and rescales the loss for each sentence,
            # sum and divide it by the sentence amount.
            # per_batch_loss: (batch, )
            # num_non_empty_sequences: (1,)
            # loss_lm_margin: (1,)
            per_batch_loss = - (pos_log_prob * target_mask.float()).sum(1) / (target_mask.sum(1).float() + 1e-20)
            loss = per_batch_loss.sum() / num_non_empty_sequences
        else:
            # neg_log_prob: (batch, length, 1) -> (batch, length)
            neg_log_prob = log_probs.gather(dim=-1, index=neg_target.unsqueeze(-1)).squeeze(-1)

            # use the margin loss as the regularization
            # neg_batch_loss: (batch,)
            # pos_batch_loss: (batch,)
            # batch_margin_loss: (batch,)
            neg_batch_loss = (neg_log_prob * neg_target_mask.float()).sum(1) / (neg_target_mask.sum(1).float() + 1e-20)
            pos_batch_loss = (pos_log_prob * target_mask.float()).sum(1) / (target_mask.sum(1).float() + 1e-20)
            batch_magin_loss = torch.relu(neg_batch_loss - pos_batch_loss + self._margin)

            # loss_lm_margin: (1,)
            loss = batch_magin_loss.sum() / num_non_empty_sequences
        return loss


    def _find_next_keyword(self,
                           last_pred: torch.LongTensor,
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

    def _build_training_target(self, tgt: torch.LongTensor, kwd: torch.LongTensor) -> torch.LongTensor:
        """
        Build a keywords input sequence based on the target and the keywords of it for training.
        :param tgt: (batch, tgt_len)
        :param kwd: (batch, constraint_len)
        :return: keyword_sequence: (batch, tgt_len)
        """
        batch_size = tgt.size()[0]

        # tgt_prime: (tgt_len, batch)
        tgt_prime = tgt.transpose(0, 1)

        # last_kwd_index: (batch,), at first, keyword indices are all starting from 0
        # keywords_by_step: [(batch,)]
        last_kwd_index = tgt.new_zeros((batch_size,))
        kwd_indices_by_step = []

        for step_tgt in tgt_prime:
            # step_tgt: (batch,)
            kwd_idx = self._find_next_keyword(step_tgt, last_kwd_index, kwd)
            kwd_indices_by_step.append(kwd_idx)
            last_kwd_index = kwd_idx

        # kwd_indices: (batch, tgt_len)
        kwd_indices = torch.stack(kwd_indices_by_step, dim=-1)
        # kwd_seq: (batch, tgt_len)
        kwd_seq = kwd.gather(dim=1, index=kwd_indices)

        return kwd_seq

    def _forward_prediction(self,
                            src_hidden: torch.Tensor,
                            src_mask: torch.LongTensor,
                            kwd_hidden: torch.Tensor,
                            kwd_mask: torch.LongTensor,
                            cond: torch.LongTensor,
                            cond_mask: torch.LongTensor,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward at test time. Predictions are generated via either greedy search or beam search
        :param src_hidden:  (batch, src_len, hidden)
        :param src_mask:    (batch, src_len)
        :param kwd_hidden:  (batch, kwd_len, hidden)
        :param kwd_mask:    (batch, kwd_len)
        :param cond:        (batch, constraint_len)
        :param cond_mask:   (batch, constraint_len)
        :return: A tuple contains predictions and logits.
                prediction: (batch, max_len)
                logits:     (batch, max_len, vocab)
        """
        if self._beam_size > 1:
            return self._forward_beam_search()
        else:
            return self._forward_greedy_search(src_hidden, src_mask, kwd_hidden, kwd_mask, cond, cond_mask)

    def _forward_greedy_search(self,
                               src_hidden: torch.Tensor,
                               src_mask: torch.LongTensor,
                               kwd_hidden: torch.Tensor,
                               kwd_mask: torch.LongTensor,
                               cond: torch.LongTensor,
                               cond_mask: torch.Tensor,
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run decoder step by step for testing or validation, with no gold tokens available.
        """
        batch_size = src_hidden.size()[0]
        # batch_start: (batch,)
        batch_start = src_mask.new_ones((batch_size,), dtype=torch.long) * self._start_id

        # step_logits: a list of logits, [(batch, seq_len, vocab_size)]
        logits_by_step = []
        # a list of predicted token ids at each step: [(batch,)]
        predictions_by_step = [batch_start]
        # last_kwd_index: (batch,), at first, keyword indices are all starting from 0
        # keywords_by_step: [(batch,)]
        cond_cursor = src_mask.new_zeros((batch_size,))
        cond_cursors_by_step = []

        for timestep in range(self._max_decoding_step):
            # step_inputs: (batch, timestep + 1), i.e., at least 1 token at step 0
            # inputs_embedding: (batch, seq_len, embedding_dim)
            # step_hidden:      (batch, seq_len, hidden_dim)
            # step_logit:       (batch, seq_len, vocab_size)
            step_inputs = torch.stack(predictions_by_step, dim=1)
            inputs_embedding = self._tgt_embedding(step_inputs)

            cond_cursor = self._find_next_keyword(predictions_by_step[-1], cond_cursor, cond)
            cond_cursors_by_step.append(cond_cursor)
            step_cond_indices = torch.stack(cond_cursors_by_step, dim=1)
            step_cond_inp = cond.gather(dim=-1, index=step_cond_indices)
            cond_embedding = self._tgt_embedding(step_cond_inp)

            decoder_inp = self._decoder_inp_mapping(torch.cat([inputs_embedding, cond_embedding], dim=-1))

            step_hidden = self._decoder(src_hidden, src_mask, kwd_hidden, kwd_mask, decoder_inp, None)
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

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert the predicted word ids into discrete tokens"""
        # predictions: (batch, max_length)
        predictions = output_dict["predictions"]
        if not isinstance(predictions, numpy.ndarray):
            predictions = predictions.detach().cpu().numpy()
        all_predicted_tokens = []

        if predictions.ndim == 2:
            predictions = numpy.expand_dims(predictions, 1)

        for beams in predictions:
            all_beams_tokens = []
            for token_ids in beams:
                token_ids = list(token_ids)
                if self._start_id in token_ids:
                    token_ids = token_ids[(token_ids.index(self._start_id) + 1):]   # skip the start_id
                if self._eos_id in token_ids:
                    token_ids = token_ids[:token_ids.index(self._eos_id)]
                tokens = [self.vocab.get_token_from_index(token_id, namespace=self._target_namespace)
                          for token_id in token_ids]
                all_beams_tokens.append(tokens)
            all_predicted_tokens.append(all_beams_tokens)
        output_dict['predicted_tokens'] = all_predicted_tokens
        return output_dict
