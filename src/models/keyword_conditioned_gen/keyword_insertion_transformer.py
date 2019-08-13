from typing import List, Optional, Tuple, Dict, Union
import torch.nn

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
                 max_decoding_step: int = 20,
                 ):
        super(KeywordInsertionTransformer, self).__init__()

        self.vocab = vocab

        self._encoder = encoder
        self._decoder = decoder
        self._src_emb = src_embedding
        self._tgt_emb = tgt_embedding

        self._start_id = vocab.get_token_index(start_symbol, target_namespace)
        self._eos_id = vocab.get_token_index(eos_symbol, target_namespace)
        self._padding_id = vocab.get_token_index(PADDING_TOKEN, target_namespace)
        self._max_decoding_step = max_decoding_step

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

    def forward(self,
                source_tokens: List[torch.LongTensor],
                decoder_input: List[torch.LongTensor],
                targets: Optional[Tuple]
                ) -> Dict:
        """

        :param source_tokens: [(src_len,)]
        :param decoder_input: [(tgt_len,)], a batch list of target tokens
        :param targets: ([(batch, slot_locs)], [(batch, slot_contents)], [(batch, slot_weights)])
        :return:
        """
        source_tokens_batch = pad_sequence(source_tokens, batch_first=True, padding_value=self._padding_id)
        src, src_mask = prepare_input_mask(source_tokens_batch)
        src_emb = self._src_emb(src)
        src_hid = self._encoder(src_emb, src_mask)

        if self.training and targets is not None:
            return self._forward_training(src_hid, src_mask, decoder_input, targets)
        else:
            return self._forward_inference()

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
            loss = ((slot_loss + word_loss) * gold_weights * target_mask / (slot_count + 1e-25)).sum()
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

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

