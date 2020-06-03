from typing import Union, Any
import torch
from torch import nn
from ..modules.word_char_embedding import WordCharEmbedding

class SeqESIM(nn.Module):
    def __init__(self,
                 a_embedding: Union[nn.Embedding, WordCharEmbedding],   # embedding layer
                 b_embedding: Union[nn.Embedding, WordCharEmbedding],
                 a_encoder,     # usually an LSTM or word embedding
                 b_encoder,
                 alignment,
                 a_precomp_trans,
                 b_precomp_trans,
                 a_composition,
                 b_composition,
                 a_pooling,
                 b_pooling,
                 prediction,
                 dropout: float = 0.2,
                 padding_a: int = 0,    # if you use trialbot.data.NSVocabulary, padding value is 0
                 padding_b: int = 0,
                 padding_a_char: int = 0,
                 padding_b_char: int = 0,
                 use_char_emb: bool = False,
                 ):
        super().__init__()
        self.a_embedding = a_embedding
        self.b_embedding = b_embedding
        self.a_encoder = a_encoder
        self.b_encoder = b_encoder
        self.alignment = alignment
        self.a_composition = a_composition
        self.b_composition = b_composition
        self.a_precomp_trans = a_precomp_trans
        self.b_precomp_trans = b_precomp_trans
        self.a_pooling = a_pooling
        self.b_pooling = b_pooling
        self.prediction = prediction
        self.paddings = (padding_a, padding_b, padding_a_char, padding_b_char)
        self.dropout = nn.Dropout(dropout)
        self.char_mode = use_char_emb

    def forward(self, *input: Any, **kwargs: Any):
        if self.char_mode:
            return self.forward_word_char_tokens(*input, **kwargs)
        else:
            return self.forward_word_tokens(*input, **kwargs)

    def forward_word_tokens(self, word_a: torch.Tensor, word_b: torch.Tensor):
        a_emb = self.a_embedding(word_a)
        b_emb = self.b_embedding(word_b)
        mask_a, mask_b = list(map(lambda x, y: (x != y).long(),
                                  (word_a, word_b), (self.padding_a, self.padding_a)))
        return self.forward_embs(a_emb, b_emb, mask_a, mask_b)

    def forward_word_char_tokens(self, word_a, word_b, char_a, char_b):
        get_mask_fn = lambda x, pad: (x != pad).long()

        masks = list(map(get_mask_fn, (word_a, word_b, char_a, char_b), self.paddings))

        mask_a, mask_b, mask_a_char, mask_b_char = masks

        a = self.a_embedding(word_a, char_a, mask_a_char)
        b = self.b_embedding(word_b, char_b, mask_b_char)
        return self.forward_embs(a, b, mask_a, mask_b)

    def forward_embs(self, a, b, a_mask, b_mask):
        a = self.a_encoder(a, a_mask)
        b = self.b_encoder(b, b_mask)

        align_a, align_b = self.alignment(a, b, a_mask, b_mask)

        a = torch.cat([a, align_a, a - align_a, align_a - a, a * align_a], dim=-1)
        b = torch.cat([b, align_b, b - align_b, align_b - b, b * align_b], dim=-1)

        a = self.a_precomp_trans(a)
        b = self.b_precomp_trans(b)

        a = self.a_composition(a, a_mask)
        b = self.b_composition(b, b_mask)

        a = self.a_pooling(a, a_mask)
        b = self.b_pooling(b, b_mask)

        logits = self.prediction(a, b)
        return logits

