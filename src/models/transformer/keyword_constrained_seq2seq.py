from typing import Dict, List, Tuple, Mapping, Optional

import numpy
import random
import torch
import torch.nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TokenEmbedder
from allennlp.nn import util
from .parallel_seq2seq import ParallelSeq2Seq, sequence_cross_entropy_with_probs

class KeywordConstrainedTransformer(ParallelSeq2Seq):
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
                 output_is_logit = True,
                 beam_size: int = 1,
                 diversity_factor: float = 0.,
                 accumulation_factor: float = 1.,
                 margin: float = 1.,
                 alpha: float = 1.,
                 ):
        super(KeywordConstrainedTransformer, self).__init__(
            vocab, encoder, decoder, source_embedding, target_embedding, target_namespace,
            start_symbol, eos_symbol, max_decoding_step, use_bleu, label_smoothing,
            output_projection_layer, output_is_logit,
            beam_size, diversity_factor, accumulation_factor,
        )

        self._margin = margin
        self._alpha = alpha

    def forward(self,
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None,
                keyword_tokens: Dict[str, torch.LongTensor] = None):

        source, source_mask = self.prepare_input(source_tokens)
        target, target_mask = self.prepare_input(target_tokens)

        state = self._encode(source, source_mask)

        if target is not None and self.training:
            assert keyword_tokens is not None
            keyword, keyword_mask = self.prepare_input(keyword_tokens)
            # training: use keyword for loss computation
            predictions, logits = self._forward_training(state, target[:, :-1], source_mask, target_mask[:, :-1])
            self.compute_metric(predictions, target)
            loss = self.get_training_loss(predictions, logits, target, target_mask, keyword, keyword_mask)

        elif target is not None: # validation, requires model in eval mode
            # validation: just BLEU evaluation
            predictions, logits = self._forward_greedy_search(state, source_mask)
            self.compute_metric(predictions, target)
            loss = None

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

    def compute_metric(self, predictions, target):
        if self._bleu:
            self._bleu(predictions, target[:, 1:])

    def prepare_input(self, tokens):
        if tokens is not None:
            token_ids, mask = tokens['tokens'], util.get_text_field_mask(tokens)
        else:
            token_ids, mask = None, None
        return token_ids, mask

    def get_training_loss(self,
                          predictions: torch.Tensor,
                          logits: torch.Tensor,
                          target: torch.LongTensor,
                          target_mask: torch.LongTensor,
                          keyword: torch.LongTensor,
                          keyword_mask: torch.LongTensor):
        # predictions: (batch, target_length)
        # target: (batch, target_length), containing the output IDs
        # target_mask: (batch, target_length), containing the output IDs

        target_len = target.size()[1] - 1
        pred_len = logits.size()[1]
        max_len = min(target_len, pred_len)

        # logits: (batch, length, vocab)
        # target: (batch, length)
        # target_mask: (batch, length)
        logits = logits[:, :max_len, :].contiguous()
        target = target[:, 1:(max_len + 1)].contiguous()
        target_mask = target_mask[:, 1:(max_len + 1)].float().contiguous()

        # log_prob: (batch, length, vocab)
        log_probs = torch.log_softmax(logits, -1)

        # ==================== loss 1: LM as the regularizer ==================
        # construct negative samples
        batch_size = target.size()[0]
        shift_range = random.randint(1, batch_size // 2)
        neg_target = torch.cat((target[-shift_range:], target[:-shift_range]))
        neg_target_mask = torch.cat((target_mask[-shift_range:], target_mask[:-shift_range]))

        # pos_log_prob: (batch, length)
        # neg_log_prob: (batch, length)
        pos_log_prob = log_probs.gather(dim=-1, index=target.unsqueeze(-1)) * target_mask.unsqueeze(-1)
        neg_log_prob = log_probs.gather(dim=-1, index=neg_target.unsqueeze(-1)) * neg_target_mask.unsqueeze(-1)

        if self._alpha < 0:
            loss_lm_margin = pos_log_prob.sum(1).mean(0)
        else:
            loss_lm_margin = torch.relu(neg_log_prob.sum(1) - pos_log_prob.sum(1) + self._margin).mean()

        # ==================== loss 2: keyword MLE ====================
        # keyword: (batch, keyword)
        # keyword_mask: (batch, keyword)
        keyword_sz = keyword.size()[1]
        if keyword_sz <= 0: # no keyword, then no more keyword loss
            return self._alpha * loss_lm_margin

        # expand_keyword: (batch, keyword, length, 1)
        # expand_logprob: (batch, keyword, length, vocab)
        expand_keyword = keyword.reshape(batch_size, keyword_sz, 1, 1).expand(-1, -1, max_len, -1)
        expand_logprob = log_probs.reshape(batch_size, 1, max_len, -1).expand(-1, keyword_sz, -1, -1)

        # keyword_logprob: (batch, keyword, length, 1) -> (batch, keyword, length)
        # expand_target_mask: (batch, keyword, length)
        keyword_logprob = expand_logprob.gather(dim=-1, index=expand_keyword).squeeze(-1)
        expand_target_mask = (target_mask.unsqueeze(1) + 1e-45).log().expand(-1, keyword_sz, -1)

        # when choosing the max value, the masked positions are almost negative infinities, which will get lost
        # keyword_max_logprob: (batch, keyword)
        keyword_max_logprob, _ = (keyword_logprob + expand_target_mask).max(dim=-1)

        # again, only at least one keyword (assumed length 2) is required
        # max_keyword_max_logprob: (batch, keyword)
        log_keyword_mask = (keyword_mask.float() + 1e-45).log_()
        max_keyword_max_logprob, _ = (keyword_max_logprob + log_keyword_mask).topk(2, dim=-1)

        loss_keyword = -max_keyword_max_logprob.sum(1).mean()

        # ============== final loss ================

        loss = self._alpha * loss_lm_margin + loss_keyword

        return loss

