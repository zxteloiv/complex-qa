from typing import Dict, List, Tuple, Optional
# denote the type that can be None, but must be specified and does not come with a default value.
from models.interfaces.encoder import EmbedAndGraphEncode, EmbedAndEncode
import torch
import torch.nn as nn
from trialbot.data.ns_vocabulary import NSVocabulary
from allennlp.training.metrics import BLEU, Perplexity, Average
from utils.nn import prepare_input_mask, seq_cross_ent
from utils.seq_collector import SeqCollector
from utils.text_tool import make_human_readable_text

Nullable = Optional


class ParallelSeq2Seq(nn.Module):
    def __init__(self,
                 vocab: NSVocabulary,
                 embed_encoder: EmbedAndEncode,
                 decoder: torch.nn.Module,
                 target_embedding: nn.Embedding,
                 word_projection: nn.Module,
                 src_namespace: str,
                 tgt_namespace: str,
                 start_id: int,
                 end_id: int,
                 pad_id: int = 0,
                 max_decoding_step: int = 50,
                 beam_size: int = 1,
                 diversity_factor: float = 0.,
                 accumulation_factor: float = 1.,
                 use_bleu: bool = False,
                 flooding_bias: float = -1,
                 ):
        super().__init__()
        self.vocab = vocab
        self._embed_encoder = embed_encoder
        self._decoder = decoder
        self._tgt_embedding = target_embedding

        self.start_id = start_id
        self.eos_id = end_id
        self.pad_id = pad_id
        self.max_decoding_step = max_decoding_step

        self.src_namespace = src_namespace
        self.tgt_namespace = tgt_namespace

        self._word_proj = word_projection
        self.bleu = None
        if use_bleu:
            self.bleu = BLEU(exclude_indices={pad_id, start_id, end_id})

        self.flooding_bias = flooding_bias

        self.ppl = Perplexity()
        self.err = Average()
        self.src_len = Average()
        self.tgt_len = Average()
        self.item_count = 0

        self._beam_size = beam_size
        self._diversity_factor = diversity_factor
        self._acc_factor = accumulation_factor

    def forward(self,
                source_tokens: torch.LongTensor,
                target_tokens: Optional[torch.LongTensor] = None,
                **kwargs,
                ) -> dict:
        """Run the network, and dispatch work to helper functions based on the runtime"""
        # source: (batch, source_length), containing the input word IDs
        # target: (batch, target_length), containing the output IDs

        if kwargs.get('source_graph') is not None:
            assert isinstance(self._embed_encoder, EmbedAndGraphEncode)
            self._embed_encoder.set_graph(kwargs['source_graph'])

        layered_states, state_mask = self._embed_encoder(source_tokens)
        state = layered_states[-1]  # transformer decoder only attends to top layer only

        output = dict()
        target, target_mask = prepare_input_mask(target_tokens, self.pad_id)
        if self.training:
            predictions, logits = self._forward_training(state, target[:, :-1],
                                                         state_mask, target_mask[:, :-1])
            loss = seq_cross_ent(logits, target[:, 1:].contiguous(), target_mask[:, 1:].float())
            if self.flooding_bias > 0:
                loss = (loss - self.flooding_bias).abs() + self.flooding_bias
            output['loss'] = loss

        else:
            if target_tokens is not None:
                predictions, logits = self._forward_training(state, target[:, :-1],
                                                             state_mask, target_mask[:, :-1])
            else:
                predictions, logits = self._forward_prediction(state, state_mask, decoding_len=None)

        if target_tokens is not None:
            self._compute_metrics(logits, predictions, target[:, 1:].contiguous(), target_mask[:, 1:].float())

        output.update(predictions=predictions, source=source_tokens, target=target_tokens)
        return output

    def _compute_metrics(self, logit, prediction, target, target_mask) -> None:
        """
        :param logit: (batch, seq, #vocab)
        :param prediction: (batch, seq)
        :param target: (batch, seq)
        :param target_mask: (batch, seq)
        :return:
        """
        if self.bleu is not None:
            self.bleu(prediction, target)

        xent = seq_cross_ent(logit, target, target_mask, average="batch")
        self.ppl(xent)

        # err rate must be strictly reported, thus the difference between batch sizes is important
        err = ((prediction != target) * target_mask).sum(-1) > 0
        for instance_err in err:
            self.err(instance_err)

        for l in target_mask.sum(1):
            self.tgt_len(l)

        self.item_count += prediction.size()[0]

    def revert_tensor_to_string(self, output_dict: dict) -> dict:
        """Convert the predicted word ids into discrete tokens"""
        # predictions: (batch, max_length)
        output_dict['predicted_tokens'] = make_human_readable_text(
            output_dict['predictions'], self.vocab, self.tgt_namespace,
            stop_ids=[self.eos_id, self.pad_id]
        )
        output_dict['source_tokens'] = make_human_readable_text(
            output_dict['source'], self.vocab, self.src_namespace,
            stop_ids=[self.eos_id, self.pad_id]
        )
        output_dict['target_tokens'] = make_human_readable_text(
            output_dict['target'], self.vocab, self.tgt_namespace,
            stop_ids=[self.eos_id, self.pad_id]
        )
        return output_dict

    def _encode(self, source: torch.LongTensor, source_mask: torch.LongTensor) -> torch.Tensor:
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
                          state_mask: Nullable[torch.LongTensor],
                          target_mask: Nullable[torch.LongTensor]
                          ) -> Tuple[torch.Tensor, torch.FloatTensor]:
        """
        Run decoder for training, given target tokens as supervision.
        When training, all timesteps are used and computed universally.
        """
        # target_embedding: (batch, max_target_length, embedding_dim)
        # target_hidden:    (batch, max_target_length, hidden_dim)
        # logits:           (batch, max_target_length, vocab_size)
        # predictions:      (batch, max_target_length)
        target_embedding = self._tgt_embedding(target)
        target_hidden = self._decoder(target_embedding, target_mask, state, state_mask)
        logits = self._word_proj(target_hidden)
        predictions = torch.argmax(logits, dim=-1)

        return predictions, logits

    def _forward_prediction(self, state, source_mask, *, decoding_len: Nullable[int]):
        if self._beam_size > 1:
            return self._forward_beam_search(state, source_mask, decoding_len=decoding_len)
        else:
            return self._forward_greedy_search(state, source_mask, decoding_len=decoding_len)

    def _forward_greedy_search(self,
                               state: torch.Tensor,
                               source_mask: Nullable[torch.LongTensor],
                               *,
                               decoding_len: Nullable[int],
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run decoder step by step for testing or validation, with no gold tokens available.
        """
        batch_size = state.size()[0]
        # batch_start: (batch,)
        batch_start = source_mask.new_full((batch_size,), self.start_id, dtype=torch.long)
        decoding_len = decoding_len or self.max_decoding_step

        mem = SeqCollector()
        mem(preds=batch_start)
        for timestep in range(decoding_len):
            # step_inputs: (batch, timestep + 1), i.e., at least 1 token at step 0
            # inputs_embedding: (batch, seq_len, embedding_dim)
            # step_hidden:      (batch, seq_len, hidden_dim)
            # step_logit:       (batch, seq_len, vocab_size)
            step_inputs = mem.get_stacked_tensor('preds')
            inputs_embedding = self._tgt_embedding(step_inputs)
            step_hidden = self._decoder(inputs_embedding, None, state, source_mask)
            step_logit = self._word_proj(step_hidden)

            # a list of logits, [(batch, vocab_size)]
            # prediction: (batch, seq_len)
            # greedy decoding
            mem(logits=step_logit[:, -1, :], preds=step_logit.argmax(dim=-1)[:, -1])
            del step_inputs, inputs_embedding, step_hidden, step_logit

        # predictions: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        # <start> token must not be in the predictions to keep output semantic consistent,
        # although it is implemented as resided in the prediction list for the sake of the api brevity.
        predictions = mem.get_stacked_tensor('preds')[:, 1:]
        logits = mem.get_stacked_tensor('logits')
        del mem
        return predictions, logits

    def _forward_beam_search(self,
                             state: torch.Tensor,
                             source_mask: Nullable[torch.LongTensor],
                             *,
                             decoding_len: Nullable[int],
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sz, beam_sz, vocab_sz = state.size()[0], self._beam_size, self._vocab_size
        decoding_len = decoding_len or self.max_decoding_step

        # batch_start: (batch, 1)
        # inputs_embedding: (batch, 1, embedding_dim)
        # step_hidden:      (batch, 1, hidden_dim)
        # step_logit:       (batch, 1, vocab)
        batch_start = source_mask.new_ones((batch_sz, 1), dtype=torch.long) * self._start_id
        inputs_embedding = self._tgt_embedding(batch_start)
        step_hidden = self._decoder(inputs_embedding, None, state, source_mask)
        step_logit = self._word_proj(step_hidden).squeeze(1)

        # topk_value, topk_indices: (batch, beam)
        # last_preds: (batch, beam), start tokens
        topk_value, topk_indices = step_logit.log_softmax(-1).topk(beam_sz, dim=-1)

        # diversity_value: (batch, 1, beam)
        # acc_log_probs: (batch, beam)
        diversity_value = topk_value.new_ones((batch_sz, beam_sz)).cumsum(dim=1).unsqueeze(1)
        acc_log_probs = topk_value

        # source_mask: (batch, max_input_length) -> (batch * beam, max_input_length)
        # states: (batch, max_input_length, hidden_dim) -> (batch * beam, max_input_length, hidden_dim)
        source_mask = source_mask.expand(batch_sz * beam_sz, -1)
        state = state.expand(batch_sz * beam_sz, -1, -1)

        # [(batch, beam)]
        prev_preds = [batch_start.expand(-1, beam_sz), topk_indices]
        # [(batch, beam)]
        backtrace: List[torch.Tensor] = [torch.zeros_like(topk_indices)]

        for timestep in range(decoding_len):
            # step_inputs: (batch * beam, seq_len = (timestep + 1)), i.e., at least 1 token at step 0
            step_inputs = self._backtrace_predictions(prev_preds, backtrace).reshape(batch_sz * beam_sz, -1)

            # inputs_embedding: (batch * beam, seq_len, embedding_dim)
            # step_hidden:      (batch * beam, hidden_dim)
            # step_logit:       (batch * beam, vocab)
            # step_log_probs:   (batch * beam, vocab)
            inputs_embedding = self._tgt_embedding(step_inputs)
            step_hidden = self._decoder(inputs_embedding, None, state, source_mask)[:, -1, :]
            step_logit: torch.Tensor = self._word_proj(step_hidden)

            # transition_log_probs: (batch, beam, vocab_size)
            # topk_trans_val, topk_trans_idx: (batch, beam, beam)
            transition_log_probs = step_logit.log_softmax(-1).reshape(batch_sz, beam_sz, vocab_sz)
            topk_trans_val, topk_trans_idx = transition_log_probs.topk(beam_sz, dim=-1)

            # decision_scores: (batch, beam * beam)
            decision_scores = self._acc_factor * acc_log_probs.unsqueeze(-1)
            decision_scores = decision_scores + topk_trans_val - self._diversity_factor * diversity_value
            decision_scores = decision_scores.reshape(batch_sz, beam_sz * beam_sz)

            # topk_value: (batch, beam)
            # topk_indices: (batch, beam)
            topk_value, topk_indices = decision_scores.topk(beam_sz)

            # beam_id: (batch, beam)
            # id_to_word_id: (batch, beam)
            beam_id = topk_indices // beam_sz
            # id_to_word_id = topk_indices % vocab_sz

            backtrace.append(beam_id)
            prev_preds.append(topk_trans_idx.reshape(batch_sz, beam_sz * beam_sz).gather(dim=-1, index=topk_indices))
            acc_log_probs = topk_value

        # predictions: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        predictions = self._backtrace_predictions(prev_preds, backtrace)[:, 1:]
        return predictions, acc_log_probs

    def _backtrace_predictions(self,
                               preds: List[torch.Tensor],
                               backtrace: List[torch.Tensor]) -> torch.Tensor:
        """
        :param preds: [(batch, beam)]
        :param backtrace: [(batch, beam)] with the less length than predictions by 1
        :return:
        """
        assert len(backtrace) == len(preds) - 1
        new_preds = [preds[-1]]

        last_trace = None
        for pred, trace in zip(reversed(preds[:-1]), reversed(backtrace)):
            # pred, trace: (batch, beam)
            if last_trace is not None:
                trace = trace.gather(index=last_trace, dim=-1)

            new_preds.append(pred.gather(index=trace, dim=-1))
            last_trace = trace

        return torch.stack(list(reversed(new_preds)), dim=-1)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self.bleu:
            all_metrics.update(self.bleu.get_metric(reset=reset))
        all_metrics.update(PPL=self.ppl.get_metric(reset))
        all_metrics.update(ERR=self.err.get_metric(reset))
        all_metrics.update(SLEN=self.src_len.get_metric(reset))
        all_metrics.update(TLEN=self.tgt_len.get_metric(reset))
        all_metrics.update(COUNT=self.item_count)
        if reset:
            self.item_count = 0
        return all_metrics

    get_metric = get_metrics

