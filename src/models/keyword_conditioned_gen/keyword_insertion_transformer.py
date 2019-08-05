from typing import List, Optional, Tuple, Dict
import torch.nn
import random
import numpy

from allennlp.modules import TokenEmbedder
from allennlp.training.metrics import BLEU
from utils.nn import add_positional_features, prepare_input_mask
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
                 output_projection: Optional[torch.nn.Module] = None,
                 output_is_logit: bool = True,
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

        self._output_projection = output_projection
        self._output_is_logit = output_is_logit
        if output_projection is None:
            self._output_projection_layer = torch.nn.Linear(decoder.hidden_dim, self._vocab_size)
            self._output_is_logit = True

    def forward(self,
                source_tokens: List[torch.LongTensor],
                src_keyword_tokens: List[torch.LongTensor],
                target_tokens: Optional[List[torch.LongTensor]],
                tgt_keyword_tokens: List[torch.LongTensor],
                ) -> Dict:
        """

        :param source_tokens: [(src_len,)]
        :param src_keyword_tokens: [(skwd_len,)], not used so far
        :param target_tokens: [(tgt_len,)], a batch list of target tokens
        :param tgt_keyword_tokens: [(tkwd_len,)]
        :return:
        """
        source_tokens_batch = pad_sequence(source_tokens, batch_first=True, padding_value=self._padding_id)
        src, src_mask = prepare_input_mask(source_tokens_batch)
        src_emb = self._src_embedding(src)
        src_hid = self._src_enc(src_emb, src_mask)

        if target_tokens is None:   # testing mode
            return self._forward_inference(src_hid, src_mask, tgt_keyword_tokens)

        tgt, tgt_mask = prepare_input_mask(target_tokens)
        tkwd, tkwd_mask = prepare_input_mask(tgt_keyword_tokens)

        if target_tokens is not None and self.training: # training mode
            return self._forward_training(src_hid, src_mask, tgt, tkwd)
        elif target_tokens is not None: # validation mode
            return self._forward_validation(src_hid, src_mask, tgt, tkwd)

    def _forward_training(self,
                          src_hid: torch.Tensor,
                          src_mask: torch.LongTensor,
                          tgt_toks: Optional[List[torch.LongTensor]],
                          tkwd_toks: Optional[List[torch.LongTensor]],
                          ):
        """
        Run the forward pass during training.

        :param src_hid: (batch, src, hidden)
        :param src_mask: (batch, src)
        :param skwd: (batch, skwd)
        :param skwd_mask: (batch, skwd)
        :param tgt: (batch, target)
        :param tgt_mask: (batch, target)
        :param tkwd: (batch, tkwd)
        :param tkwd_mask: (batch, tkwd)
        :return:
        """
        # tgt_hid: (batch, target, hidden)
        # dec_out: (batch, target - 1, hidden)
        tgt_hid = self._tgt_emb(tgt)
        dec_out = self._decoder(tgt_hid, tgt_mask, src_hid, src_mask)
