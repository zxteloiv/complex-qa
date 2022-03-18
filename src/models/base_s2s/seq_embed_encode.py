from typing import Tuple, List

import torch.nn

from models.interfaces.encoder import EmbedAndEncode, EncoderStack
from models.modules.variational_dropout import VariationalDropout
from utils.nn import prepare_input_mask


class SeqEmbedEncoder(EmbedAndEncode):
    def __init__(self,
                 source_embedding: torch.nn.Embedding,
                 encoder: EncoderStack,
                 padding_index: int = 0,
                 enc_dropout: float = 0.,
                 ):
        super().__init__()
        self._src_embedding = source_embedding
        self._encoder = encoder
        self._padding_index = padding_index
        self._src_emb_dropout = VariationalDropout(enc_dropout, on_the_fly=True)

    def forward(self, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # source: (batch, source_length), containing the input word IDs
        # target: (batch, target_length), containing the output IDs
        source, source_mask = prepare_input_mask(tokens, self._padding_index)
        source_embedding = self.embed(tokens)
        layered_hidden = self.encode(source_embedding, source_mask)

        # by default, state_mask is the same as the source mask,
        # but if the state is not directly aligned with source,
        # it will be different.
        state_mask = source_mask
        return layered_hidden, state_mask

    def embed(self, source):
        # source: (batch, max_input_length), source sequence token ids
        # source_mask: (batch, max_input_length), source sequence padding mask
        # source_embedding: (batch, max_input_length, embedding_sz)
        source_embedding = self._src_embedding(source)
        source_embedding = self._src_emb_dropout(source_embedding)
        return source_embedding

    def encode(self, source_embedding, source_mask):
        self._encoder(source_embedding, source_mask)
        layered_hidden = self._encoder.get_layered_output()
        return layered_hidden

    def is_bidirectional(self):
        return self._encoder.is_bidirectional()

    def get_output_dim(self) -> int:
        return self._encoder.get_output_dim()

