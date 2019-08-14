from typing import Dict, List, Tuple, Mapping, Optional
import torch
import torch.nn

from utils.nn import add_positional_features

from .multi_head_attention import MultiHeadAttention, MultiHeadSelfAttention

class InsertionDecoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,  # input embedding dimension
                 num_layers: int = 6,
                 num_heads: int = 8,
                 feedforward_hidden_dim: int = None,
                 feedforward_dropout: float = 0.1,
                 attention_dim: int = None,
                 value_dim: int = None,
                 residual_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_positional_embedding: bool = True,
                 ):
        """
        Construct a decoder for insertion transformer,
        which is in charge of modules in the transformer model
        from the Positional Embedding before the final linear projection.
        The embedding and linear projection should be implemented elsewhere.

        In contrast with the vanilla transformer decoder, differences are listed as follows.
        1. The insertion decoder DOES NOT use a masked self attention,
        therefore the implementation must not reuse a vanilla transformer decoder.
        2. In addition, the adjacent output are concatenated to produce the slot representation.

        :param input_dim: int, number of the input dimension, and assumed to be the output dimension, too.
        :param num_layers: int, number of stacked layers
        :param num_heads: int, number of heads in multi-head attention
        :param feedforward_hidden_dim: int, number of dimension of the feedforward layers, equals input_dim if not given
        :param feedforward_dropout: float, dropout ratio after the feedforward hidden layer
        :param attention_dim: int, attention dimension of the attention, assumed input_dim/num_heads if not given
        :param value_dim: int, value dimension of the attention, assumed input_dim/num_heads if not given
        :param residual_dropout: float, dropout ratio between the residual connection every block components
        :param attention_dropout: float, dropout ratio for multihead attention
        :param use_positional_embedding: bool, whether use positional embedding before the first decoder block
        """
        super(InsertionDecoder, self).__init__()

        hidden_dim = input_dim  # the hidden states dimension outputted by the decoder module
        attention_dim = attention_dim or (hidden_dim // num_heads)
        value_dim = value_dim or (hidden_dim // num_heads)
        feedforward_hidden_dim = feedforward_hidden_dim or hidden_dim

        class _DecoderBlock(torch.nn.Module):
            def __init__(self):
                super(_DecoderBlock, self).__init__()
                self._self_attn = MultiHeadSelfAttention(
                    num_heads,
                    hidden_dim,
                    attention_dim * num_heads,
                    value_dim * num_heads,
                    attention_dropout=attention_dropout,
                )
                self._self_attn_norm = torch.nn.LayerNorm(hidden_dim)
                self._attn = MultiHeadAttention(
                    num_heads,
                    hidden_dim,
                    hidden_dim,
                    attention_dim * num_heads,
                    value_dim * num_heads,
                    attention_dropout=attention_dropout
                )
                self._attn_norm = torch.nn.LayerNorm(hidden_dim)
                self._feedforward = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, feedforward_hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(feedforward_dropout),
                    torch.nn.Linear(feedforward_hidden_dim, hidden_dim),
                )
                self._feedforward_norm = torch.nn.LayerNorm(hidden_dim)
                self._dropout = torch.nn.Dropout(residual_dropout)

            def forward(self, src, src_mask, inp, inp_mask):
                """
                Run forward pass of the decoder block of the Transformer Model

                :param src: (batch, src, hidden)
                :param src_mask: (batch, src)
                :param inp: (batch, tgt, hidden)
                :param inp_mask: (batch, tgt, hidden)
                :return:
                """
                self_attn_out, _ = self._self_attn(inp, inp_mask)
                self_attn_out = self._dropout(self_attn_out)
                self_attn_out = self._self_attn_norm(self_attn_out + inp)

                attn_out, _ = self._attn(self_attn_out, src, src_mask)
                attn_out = self._dropout(attn_out)
                attn_out = self._attn_norm(attn_out + self_attn_out)

                feedforward_out = self._feedforward(attn_out)
                feedforward_out = self._dropout(feedforward_out)
                feedforward_out = self._feedforward_norm(feedforward_out)

                return feedforward_out


        self._dec_blocks = torch.nn.ModuleList([ _DecoderBlock() for _ in range(num_layers) ])
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2
        self._use_positional_embedding = use_positional_embedding

    def forward(self,
                tgt_hidden: torch.Tensor,
                tgt_mask: Optional[torch.LongTensor],
                src_hidden: torch.Tensor,
                src_mask: Optional[torch.LongTensor]
                ) -> Tuple[torch.Tensor, Optional[torch.LongTensor]]:
        """
        Transformer deocder stacked blocks.

        :param tgt_hidden: (batch, tgt, hidden)
        :param tgt_mask: (batch, tgt), 0 or 1 as a mask matrix
        :param src_hidden: (batch, src, hidden)
        :param src_mask: (batch, src)
        :return: (batch, tgt - 1, hidden)
        """
        inp = add_positional_features(tgt_hidden) if self._use_positional_embedding else tgt_hidden

        for dec_block in self._dec_blocks:
            inp = dec_block(src_hidden, src_mask, inp, tgt_mask)

        # inp is (batch, tgt_len, hidden)
        # every adjacent pair of hidden inputs are concatenated, forming a bigger but short representation for each slot
        # output is thus (batch, tgt_len - 1, hidden * 2)

        left = inp[:, :-1, :]
        right = inp[:, 1:, :]
        output = torch.cat([left, right], dim=-1)

        output_mask = None if tgt_mask is None else ((tgt_mask[:, :-1] + tgt_mask[:, 1:]) == 2).long()

        return output, output_mask

