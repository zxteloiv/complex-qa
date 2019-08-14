from typing import List, Optional, Tuple, Dict, Union
import torch.nn

import numpy as np
from allennlp.modules import TokenEmbedder
from allennlp.training.metrics import BLEU
from utils.nn import prepare_input_mask
from torch.nn.utils.rnn import pad_sequence

from ..transformer.encoder import TransformerEncoder
from ..transformer.insertion_decoder import InsertionDecoder

from trialbot.data.ns_vocabulary import NSVocabulary, PADDING_TOKEN

class KeywordInsertionTransformer(torch.nn.Module):
    def __init__(self,
                 vocab: NSVocabulary,
                 encoder: TransformerEncoder,
                 decoder: InsertionDecoder,
                 src_embedding: TokenEmbedder,
                 tgt_embedding: TokenEmbedder,
                 joint_projection: Union[torch.nn.Module, Tuple[torch.nn.Module, torch.nn.Module]],
                 vocab_bias_mapper: Optional[torch.nn.Module],
                 use_bleu: bool = True,
                 target_namespace: str = "tokens",
                 start_symbol: str = '<GO>',
                 eos_symbol: str = '<EOS>',
                 span_ends_symbol: str = '<EndOfSpan>',
                 max_decoding_step: int = 20,
                 span_end_penalty: float = .5,
                 slot_transform: Optional[TransformerEncoder] = None,
                 ):
        super(KeywordInsertionTransformer, self).__init__()

        self.vocab = vocab

        self._encoder = encoder
        self._decoder = decoder
        self._src_emb = src_embedding
        self._tgt_emb = tgt_embedding
        self._slot_trans = slot_transform

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._padding_id = vocab.get_token_index(PADDING_TOKEN, target_namespace)
        self._max_decoding_step = max_decoding_step
        self._span_end_id = vocab.get_token_index(span_ends_symbol, target_namespace)

        self._target_namespace = target_namespace
        self._vocab_size = vocab.get_vocab_size(target_namespace)

        self._bleu = None
        if use_bleu:
            pad_index = self.vocab.get_token_index(PADDING_TOKEN, target_namespace)
            self._bleu = BLEU(exclude_indices={pad_index, self._eos_id, self._start_id})

        if isinstance(joint_projection, tuple):
            self._use_joint_proj = False
            slot_proj, word_proj = joint_projection
            self._slot_proj = slot_proj
            self._word_proj = word_proj
        else:
            self._use_joint_proj = True
            self._joint_proj = joint_projection

        self._vocab_bias_mapper = vocab_bias_mapper
        self._span_end_penalty = span_end_penalty

    def forward(self,
                source_tokens: List[torch.LongTensor],
                decoder_input: List[torch.LongTensor],
                targets: Optional[Tuple] = None,
                references: Optional[List[torch.LongTensor]] = None,
                ) -> Dict:
        """

        :param source_tokens: [(src_len,)]
        :param decoder_input: [(tgt_len,)], a batch list of target tokens
        :param targets: ([(batch, slot_locs)], [(batch, slot_contents)], [(batch, slot_weights)])
        :param references: [(batch, tgt_len)], if given, it will be evaluated at inference time
        :return:
        """
        source_tokens_batch = pad_sequence(source_tokens, batch_first=True, padding_value=self._padding_id)
        src, src_mask = prepare_input_mask(source_tokens_batch)
        src_emb = self._src_emb(src)
        src_hid = self._encoder(src_emb, src_mask)

        if self.training and targets is not None:
            return self._forward_training(src_hid, src_mask, decoder_input, targets)
        else:
            return self._forward_inference(src_hid, src_mask, decoder_input, references)

    def _forward_training(self,
                          src_hid: torch.Tensor,
                          src_mask: torch.LongTensor,
                          decoder_input: List[torch.LongTensor],
                          targets: Tuple,
                          ) -> Dict:
        """
        Run forward for training.

        :param src_hid: (batch, src, hidden)
        :param src_mask: (batch, src)
        :param decoder_input: [(inp,)]
        :param targets: ([(targets,)], [(targets,)], [(targets,)])
        :return:
        """

        # inp_hid: (batch, target + 2, hidden)
        # inp_mask: (batch, target + 2)
        # dec_out: (batch, target + 1, hidden)
        # dec_out_mask: (batch, target + 1)
        inp_hid, inp_mask = self._get_input_hidden(decoder_input)
        dec_out, dec_out_mask = self._decoder(inp_hid, inp_mask, src_hid, src_mask)

        if self._slot_trans is not None:
            dec_out = self._slot_trans(dec_out, dec_out_mask)

        vocab_bias = self._get_vocab_bias(dec_out)

        # gold_locs, gold_conts, gold_weights: (batch, target_toks)
        gold_locs, gold_conts, gold_weights = list(map(
            lambda x: pad_sequence(x, batch_first=True, padding_value=self._padding_id),
            targets
        ))
        target_mask = (gold_locs != self._padding_id).float()

        if self._use_joint_proj:
            cl_dist = self._joint_proj(dec_out, vocab_bias)
            raise NotImplementedError

        else:
            # slot_probs: (batch, target + 1, 1) -> (batch, target + 1)
            # word_probs: (batch, target + 1, vocab)
            slot_probs = self._slot_proj(dec_out).squeeze(-1)
            word_probs = self._word_proj(dec_out, vocab_bias)

            # slot_loss: (batch, target_toks)
            slot_loss = -(slot_probs.gather(dim=-1, index=gold_locs) + 1e-15).log()

            # word_probs_for_each_slot: (batch, target_toks, vocab)
            # target_word_probs: (batch, target_toks, 1) -> (batch, target_toks)
            word_probs_for_each_slot = batched_index_select(word_probs, 1, gold_locs)
            target_word_probs = word_probs_for_each_slot.gather(dim=-1, index=gold_conts.unsqueeze(-1)).squeeze()
            word_loss = -(target_word_probs + 1e-15).log()

            slot_count = (inp_mask.sum(1) - 1).unsqueeze(-1).float()
            loss = ((slot_loss + word_loss) * gold_weights * target_mask / (slot_count + 1e-25)).sum(1).mean(0)
            output = {"loss": loss}

        return output

    def _get_input_hidden(self, decoder_input: List[torch.LongTensor]):
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

    def _get_vocab_bias(self, dec_out: torch.Tensor):
        if not self._vocab_bias_mapper:
            return None

        # dec_out: (batch, target + 1, hidden)
        # max_pool: (batch, hidden)
        # vocab_bias: (batch, vocab)
        max_pool = torch.max(dec_out, dim=1)[0]
        vocab_bias = self._vocab_bias_mapper(max_pool)
        return vocab_bias

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
        batch_size = len(dec_inp)
        # inp_hid: (batch, target + 2, hidden)
        # inp_mask: (batch, target + 2)
        # dec_out: (batch, target + 1, hidden)
        # dec_out_mask: (batch, target + 1)
        inp_hid, inp_mask = self._get_input_hidden(dec_inp)
        dec_out, dec_out_mask = self._decoder(inp_hid, inp_mask, src_hid, src_mask)
        if self._slot_trans is not None:
            dec_out = self._slot_trans(dec_out, dec_out_mask)

        vocab_bias = self._get_vocab_bias(dec_out)
        if self._use_joint_proj:
            cl_dist = self._joint_proj(dec_out, vocab_bias)
            raise NotImplementedError

        else:
            # slot_probs: (batch, target + 1, 1) -> (batch, target + 1)
            # word_probs: (batch, target + 1, vocab)
            slot_probs = self._slot_proj(dec_out).squeeze(-1) * dec_out_mask.float()
            word_probs = self._word_proj(dec_out, vocab_bias)

            # greedy_slot: (batch,)
            greedy_slot = torch.argmax(slot_probs, dim=-1)

            # word_probs_for_slot: (batch, vocab)
            word_probs_for_slot: torch.Tensor = word_probs[torch.arange(batch_size), greedy_slot]

            # add greedy penalty for EndOfSpan token, preventing early stop
            word_probs_for_slot[:, self._span_end_id] -= self._span_end_penalty

            word_probs_for_slot[:, self._padding_id] = 0
            word_probs_for_slot[:, self._start_id] = 0
            word_probs_for_slot[:, self._eos_id] = 0

            # word_probs_for_slot: (batch,)
            greedy_word = torch.argmax(word_probs_for_slot, dim=-1)

            next_inps = []
            for example, loc, word in zip(dec_inp, greedy_slot, greedy_word):
                # since in greedy mode, each iteration only predicts one word for each example of a batch
                # list.insert will be sufficient.
                inputs = example.tolist()
                idx = loc.item()
                word = loc.item()
                if word not in (self._span_end_id,): # self._start_id, self._eos_id, self._padding_id):
                    inputs.insert(idx, word)
                next_inps.append(example.new_tensor(inputs))
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

