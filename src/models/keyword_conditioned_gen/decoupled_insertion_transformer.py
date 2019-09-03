from typing import List, Optional, Tuple, Dict, Union
import torch.nn
import logging

import numpy as np
from allennlp.modules import TokenEmbedder
from allennlp.training.metrics import BLEU
from utils.nn import prepare_input_mask
from torch.nn.utils.rnn import pad_sequence

from ..transformer.encoder import TransformerEncoder
from ..transformer.insertion_decoder import InsertionDecoder

from trialbot.data.ns_vocabulary import NSVocabulary, PADDING_TOKEN

class DecoupledInsertionTransformer(torch.nn.Module):
    def __init__(self,
                 vocab: NSVocabulary,
                 encoder: TransformerEncoder,
                 slot_decoder: InsertionDecoder,
                 slot_predictor: torch.nn.Module,
                 content_decoder: InsertionDecoder,
                 word_predictor: torch.nn.Module,
                 src_embedding: TokenEmbedder,
                 tgt_embedding: TokenEmbedder,
                 use_bleu: bool = True,
                 target_namespace: str = "tokens",
                 start_symbol: str = '<s>',
                 eos_symbol: str = '</s>',
                 mask_symbol: str = '<mask>',
                 max_decoding_step: int = 20,
                 span_end_threshold: float = .5,
                 stammering_window: int = 2,

                 alpha1: float = .5,
                 alpha2: float = .5,
                 topk: int = 6,

                 # optional modules
                 slot_trans: Optional[TransformerEncoder] = None,
                 dual_model: Optional[torch.nn.Module] = None,
                 ):
        super().__init__()

        self.vocab = vocab

        self.logger = logging.getLogger(__name__)

        self._encoder = encoder
        self._slot_decoder = slot_decoder
        self._slot_trans = slot_trans
        self._slot_predictor = slot_predictor
        self._content_decoder = content_decoder
        self._word_predictor = word_predictor
        self._src_emb = src_embedding
        self._tgt_emb = tgt_embedding
        self._dual_model = dual_model

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._padding_id = vocab.get_token_index(PADDING_TOKEN, target_namespace)
        self._mask_id = vocab.get_token_index(mask_symbol, target_namespace)
        self._max_decoding_step = max_decoding_step
        self._stammering_window = stammering_window

        self._target_namespace = target_namespace
        self._vocab_size = vocab.get_vocab_size(target_namespace)

        self._bleu = None
        if use_bleu:
            pad_index = self.vocab.get_token_index(PADDING_TOKEN, target_namespace)
            self._bleu = BLEU(exclude_indices={pad_index, self._eos_id, self._start_id})

        self._span_end_threshold = span_end_threshold
        self._alpha1 = max(min(alpha1, 1.), 0.)
        self._alpha2 = max(min(alpha2, 1.), 0.)
        self._topk = topk

    def forward(self,
                source_tokens: List[torch.LongTensor],
                training_targets: Optional[Tuple] = None,
                inference_input: List[torch.LongTensor] = None,
                references: Optional[List[torch.LongTensor]] = None,
                ) -> Dict:
        """

        :param source_tokens: [(src_len,)]
        :param training_targets: (slot_dec_data, content_dec_data, dual_model_data), each is a list of tuples
                slot_dec_data: [(slot_dec_inp, slot_dec_target, slot_dec_target_weight)]
                cont_dec_data: [(cont_dec_inp, cont_dec_target_loc, cont_dec_target_word, cont_dec_target_weight)]
                dual_data: [(dual_inp, dual_target)]
        :param references: [(batch, tgt_len)], if given, it will be evaluated at inference time
        :return:
        """
        source_tokens_batch = pad_sequence(source_tokens, batch_first=True, padding_value=self._padding_id)
        src, src_mask = prepare_input_mask(source_tokens_batch)
        src_emb = self._src_emb(src)
        src_hid = self._encoder(src_emb, src_mask)

        if self.training and training_targets is not None:
            return self._forward_training(src_hid, src_mask, training_targets)
        else:
            return self._forward_inference(src_hid, src_mask, inference_input, references)

    def _forward_training(self,
                          src_hid: torch.Tensor,
                          src_mask: torch.LongTensor,
                          training_targets: Tuple,
                          ) -> Dict:
        """
        Run forward for training.

        :param src_hid: (batch, src, hidden)
        :param src_mask: (batch, src)
        :param training_targets: (slot_dec_data, content_dec_data, dual_model_data), each is a list of tuples
                slot_dec_data: [(slot_dec_inp, slot_dec_target, slot_dec_target_weight)]
                cont_dec_data: [(cont_dec_inp, cont_dec_target_loc, cont_dec_target_word, cont_dec_target_weight)]
                dual_data: [(dual_inp, dual_target)]
        :return:
        """

        slot_dec_data, cont_dec_data, dual_data = training_targets

        loss_slot_dec = self._get_slot_decoder_training_loss(src_hid, src_mask, slot_dec_data)
        loss_cont_dec = self._get_content_decoder_training_loss(src_hid, src_mask, cont_dec_data)

        a1, a2 = self._alpha1, self._alpha2

        # if self._dual_model is not None:
        #     loss_dual = self._get_dual_loss()
        #     loss = a1 * loss_slot_dec + a2 * loss_cont_dec + (1 - a1 - a2) * loss_dual
        # else:
        #     loss_dual = 0
        #     loss = a1 * loss_slot_dec + (1 - a1) * loss_cont_dec
        loss = a1 * loss_slot_dec + (1 - a1) * loss_cont_dec

        output = {"loss": loss, "loss_slot_dec": loss_slot_dec.item(), "loss_cont_dec": loss_cont_dec}

        return output

    def _get_content_decoder_training_loss(self, src_hid, src_mask, cont_dec_data):
        # cont_dec_inp: [(input_len,)], input may contain a partial sentence, as well as the inserted masks
        # cont_dec_target_loc: [(target_num,)], target_num >= input_len because a mask may contain multiple words
        # cont_dec_target_word: [(target_num,)]
        # cont_dec_target_weight: [(target_num,)]
        cont_dec_inp, cont_dec_target_loc, cont_dec_target_word, cont_dec_target_weight = zip(*cont_dec_data)

        # cont_inp_hid: (batch, target + 2, hidden)
        # cont_inp_mask: (batch, target + 2)
        # word_logprob: (batch, target + 2, vocab)
        cont_inp_hid, cont_inp_mask = self._get_input_hidden_for_decoders(cont_dec_inp)
        cont_hid = self._content_decoder(cont_inp_hid, cont_inp_mask, src_hid, src_mask)
        word_logprob = (self._word_predictor(cont_hid) + 1e-20).log()

        # target_locs: (batch, target_num)
        # target_mask: (batch, target_num)
        # target_words: (batch, target_num) -> (batch, target_num, 1)
        # target_weights: (batch, target_num) float
        target_locs, target_mask = self.batchify_tensor_list(cont_dec_target_loc)
        target_words = pad_sequence(cont_dec_target_word, batch_first=True, padding_value=self._padding_id).unsqueeze(-1)
        target_weights = pad_sequence(cont_dec_target_weight, batch_first=True, padding_value=self._padding_id).float()

        # Since every input is wrapped with additional tokens <s> and </s>,
        # every targets is right shifted in one place.
        target_locs += 1

        # word_logprob_per_slot: (batch, target_num, vocab)
        # target_word_logprobs: (batch, target_num, 1) -> (batch, target_num)
        word_logprob_per_slot = batched_index_select(word_logprob, 1, target_locs)
        target_word_logprobs = word_logprob_per_slot.gather(dim=-1, index=target_words).squeeze()
        word_loss = -target_word_logprobs * target_mask.float() * target_weights

        # Since targets corresponding to a single <mask> are already weighted,
        # final loss can be obtained via global mean directly.
        loss = word_loss.mean()
        return loss

    def _get_slot_decoder_training_loss(self, src_hid, src_mask, slot_dec_data):
        # slot_dec_inp: [(tgt,)]
        # slot_dec_target: [(tgt + 1,)]
        # slot_dec_target_weight: [(tgt + 1,)]
        slot_dec_inp, slot_dec_target, slot_dec_target_weight = zip(*slot_dec_data)

        # slot_inp_hid: (batch, target + 2, hidden)
        # slot_inp_mask: (batch, target + 2)
        # slot_dec_out: (batch, target + 1, hidden)
        # slot_dec_out_mask: (batch, target + 1)
        slot_inp_hid, slot_inp_mask = self._get_input_hidden_for_decoders(slot_dec_inp)
        slot_dec_out, slot_dec_out_mask = self._slot_decoder(slot_inp_hid, slot_inp_mask, src_hid, src_mask)

        if self._slot_trans:
            slot_dec_out = self._slot_trans(slot_dec_out, slot_dec_out_mask)

        # slot_probs: (batch, target + 1) <- (batch, target + 1, 1)
        slot_probs = self._slot_predictor(slot_dec_out).squeeze() * slot_dec_out_mask.float()

        # Valid targets are 0s and 1s, padding value must be a different one and we thus use -1.
        # target: (batch, target + 1)
        # target_mask: (batch, target + 1)
        # target_weight: (batch, target + 1) float
        target, target_mask = self.batchify_tensor_list(slot_dec_target, -1)
        # target_weight = pad_sequence(slot_dec_target_weight, batch_first=True, padding_value=self._padding_id).float()

        # compute MSE Loss with masks, which is not supported by torch.nn.functional.mse_loss.
        # loss is weighted within an example, and averaged within a batch
        # squared_error: (batch, target + 1)
        squared_error = (slot_probs - target.float()) ** 2 * target_mask.float()
        loss = (squared_error.sum(1) / target_mask.sum(1).float()).mean()
        return loss

    def batchify_tensor_list(self, l: List[torch.Tensor], padding_val=None) -> Tuple[torch.Tensor, torch.LongTensor]:
        if padding_val is None:
            padding_val = self._padding_id
        batch = pad_sequence(l, batch_first=True, padding_value=padding_val)
        return prepare_input_mask(batch, padding_val)

    def _get_input_hidden_for_decoders(self, decoder_input: List[torch.LongTensor]):
        _sample = decoder_input[0]
        start_id = _sample.new_full((1,), self._start_id)
        eos_id = _sample.new_full((1,), self._eos_id)

        # batch_inp: (batch, target + 2)
        wrapped_dec_inp = [torch.cat([start_id, x, eos_id]) for x in decoder_input]
        batch_inp = pad_sequence(wrapped_dec_inp, batch_first=True, padding_value=self._padding_id)

        # inp_hid: (batch, target + 2, hidden)
        # inp_mask: (batch, target + 2)
        inp_hid = self._tgt_emb(batch_inp)
        inp_mask = (batch_inp != self._padding_id).long()
        return inp_hid, inp_mask

    def _forward_inference(self, src_hid, src_mask, input_start: List[torch.LongTensor],
                           references: Optional[List[torch.LongTensor]] = None) -> Dict:

        dec_inp = input_start
        for l in range(self._max_decoding_step):
            dec_inp = self._inference_step(src_hid, src_mask, dec_inp)

        preds = pad_sequence(dec_inp, batch_first=True, padding_value=self._padding_id)
        if references is not None:
            refs = pad_sequence(references, batch_first=True, padding_value=self._padding_id)
            self._bleu(preds, refs)

        output = {"predictions": preds}
        return output

    def _inference_step(self, src_hid: torch.Tensor, src_mask: torch.Tensor, dec_inp: List[torch.LongTensor]):
        # inp_hid: (batch, target + 2, hidden)
        # inp_mask: (batch, target + 2)
        # slot_dec_out: (batch, target + 1, hidden)
        # slot_dec_out_mask: (batch, target + 1)
        inp_hid, inp_mask = self._get_input_hidden_for_decoders(dec_inp)
        slot_dec_out, slot_dec_out_mask = self._slot_decoder(inp_hid, inp_mask, src_hid, src_mask)

        if self._slot_trans:
            slot_dec_out = self._slot_trans(slot_dec_out, slot_dec_out_mask)

        # slot_probs: (batch, target + 1)
        # best_val: (batch,)
        # best_slot: (batch,) long
        slot_probs = self._slot_predictor(slot_dec_out).squeeze() * slot_dec_out_mask.float()
        best_val, best_slot  = torch.max(slot_probs, dim=-1)
        logging.debug("=" * 20 + " slot prediction " + "=" * 20)
        logging.debug("\n-----------------------\n".join(
            f"batch={i} select: prob={best_val[i].item()} slot={best_slot[i].item()}\n"
            f"inputs={dec_inp[i].tolist()}\n"
            f"tokens={[self.vocab.get_token_from_index(t, namespace=self._target_namespace) for t in dec_inp[i].tolist()]}\n"
            f"probs={[f'{val:.4f}' for val in slot_probs[i].tolist()]}"
            for i in range(best_val.size()[0])
        ))

        # word_dec_inp: [(filtered_batch, target + 1)]
        word_dec_inp = []
        for i, (prob, slot, ith_inp) in enumerate(zip(best_val, best_slot, dec_inp)):
            if prob >= self._span_end_threshold:
                content = ith_inp.tolist()
                content.insert(slot, self._mask_id)
                word_dec_inp.append(ith_inp.new_tensor(content))
            else:
                word_dec_inp.append(ith_inp)

        # mask_pos: (batch,)
        # mask positions for inputs without inserted masks are preserved at 0.
        # after adding a <s> token at the beginning, 0 will be meaningful and all other positions increases.
        mask_pos = (best_val > self._span_end_threshold).long() * (best_slot + 1)

        # word_probs: (batch, target + 1 + 2, vocab), with <mask>, <s> and </s>
        # topk_words: (batch, target + 1 + 2, K),
        cont_inp_hid, cont_inp_mask = self._get_input_hidden_for_decoders(word_dec_inp)
        cont_hid = self._content_decoder(cont_inp_hid, cont_inp_mask, src_hid, src_mask)
        word_probs = self._word_predictor(cont_hid)
        _, topk_words = word_probs.topk(self._topk, dim=-1)

        next_inps = []
        logging.debug("=" * 20 + " word prediction " + "=" * 20)
        for i, (example, pos, words) in enumerate(zip(dec_inp, mask_pos, topk_words)):
            # since in greedy mode, each iteration only predicts one word for each example of a batch
            # list.insert will be sufficient.
            inputs = example.tolist()

            # consider each K candidate at the position of <mask>
            choices = words[pos].tolist()
            word = None
            for candidate in choices:
                # Selection Rules: how to choose the next position and word.
                # 1. the special tokens must not be choose
                if candidate in (self._start_id, self._eos_id, self._padding_id, self._mask_id):
                    logging.debug(f'batch={i} '
                                  f'Rule 1 activated for word={candidate} '
                                  f'with prob={word_probs[i, pos, candidate].item()}')
                    continue

                # 2. the choice must not be the same words as either before and after the slot
                if candidate in inputs[max(0, pos - self._stammering_window - 1):(pos + self._stammering_window)]:
                    logging.debug(f'batch={i} '
                                  f'Rule 2 activated for word={candidate} '
                                  f'with prob={word_probs[i, pos, candidate].item()}')
                    continue

                # stop for the chosen one
                word = candidate
                break

            else:
                logging.debug('Fallback Rule activated')
                word = choices[0]

            if pos > 0: # only for sentence with <mask>
                inputs.insert(pos - 1, word)
                logging.debug(("-----------\n" if i > 0 else "") +
                              f"batch={i} selected: prob={word_probs[i, pos, word].item()}, "
                              f"word={word}, idx={pos}, original={inputs}")
            next_inps.append(example.new_tensor(inputs))

        logging.debug('%' * 60)

        return next_inps

    def get_metrics(self, reset=False):
        metrics = {}
        if self._bleu:
            metrics.update(self._bleu.get_metric(reset=reset))
        return metrics

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
            while self._padding_id in token_ids:
                token_ids.remove(self._padding_id)
            tokens = [self.vocab.get_token_from_index(token_id, namespace=self._target_namespace)
                      for token_id in token_ids]
            all_predicted_tokens.append(tokens)
        output_dict['predicted_tokens'] = all_predicted_tokens
        return output_dict

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

