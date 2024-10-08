from typing import Dict, List, Tuple, Mapping, Optional
import torch
import torch.nn

from allennlp.modules import FeedForward
from torch.nn import LayerNorm
from allennlp.nn import Activation
from utils.nn import add_positional_features, add_depth_features_to_single_position

from .multi_head_attention import MaskedMultiHeadSelfAttention, MultiHeadAttention


class TransformerDecoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,  # input embedding dimension
                 hidden_dim: int = None,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 attend_to_dim: int = None,
                 feedforward_hidden_dim: int = None,
                 feedforward_hidden_activation: str = "mish",
                 feedforward_dropout: float = 0.1,
                 attention_dim: int = None,
                 value_dim: int = None,
                 residual_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_positional_embedding: bool = True,
                 ):
        """
        Construct a decoder for transformer, which is in charge of modules in the transformer model
        from the Positional Embedding before the final linear projection.
        The embedding and linear projection should be implemented elsewhere.

        :param num_layers: the number of stack layers of the transformer block
        """
        super(TransformerDecoder, self).__init__()

        self._mask_attention_layers: List[MaskedMultiHeadSelfAttention] = []
        self._mask_attention_norm_layers: List[LayerNorm] = []
        self._attention_layers: List[MultiHeadAttention] = []
        self._attention_norm_layers: List[LayerNorm] = []
        self._feedforward_layers: List[FeedForward] = []
        self._feedforward_norm_layers: List[LayerNorm] = []

        hidden_dim = hidden_dim or input_dim  # the hidden states dimension outputted by the decoder module
        attend_to_dim = attend_to_dim or hidden_dim
        if hidden_dim != input_dim:
            self._emb_mapper = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self._emb_mapper = None

        attention_dim = attention_dim or (hidden_dim // num_heads)
        value_dim = value_dim or (hidden_dim // num_heads)
        feedforward_hidden_dim = feedforward_hidden_dim or hidden_dim

        for i in range(num_layers):
            masked_attention = MaskedMultiHeadSelfAttention(num_heads,
                                                            hidden_dim,
                                                            attention_dim * num_heads,
                                                            value_dim * num_heads,
                                                            attention_dropout=attention_dropout)
            self.add_module(f'masked_attention_{i}', masked_attention)
            self._mask_attention_layers.append(masked_attention)

            masked_attention_norm = LayerNorm(hidden_dim)
            self.add_module(f'masked_attention_norm_{i}', masked_attention_norm)
            self._mask_attention_norm_layers.append(masked_attention_norm)

            attention = MultiHeadAttention(num_heads,
                                           hidden_dim,
                                           attend_to_dim,
                                           attention_dim * num_heads,
                                           value_dim * num_heads,
                                           attention_dropout=attention_dropout)
            self.add_module(f'attention_{i}', attention)
            self._attention_layers.append(attention)

            attention_norm = LayerNorm(hidden_dim)
            self.add_module(f'attention_norm_{i}', attention_norm)
            self._attention_norm_layers.append(attention_norm)

            feedfoward = FeedForward(hidden_dim,
                                     num_layers=2,
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     activations=[Activation.by_name(feedforward_hidden_activation)(),
                                                  Activation.by_name('linear')()],
                                     dropout=feedforward_dropout)
            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedforward_layers.append(feedfoward)

            feedforward_norm = LayerNorm(hidden_dim)
            self.add_module(f"feedforward_norm_{i}", feedforward_norm)
            self._feedforward_norm_layers.append(feedforward_norm)

        self._dropout = torch.nn.Dropout(residual_dropout)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._use_positional_embedding = use_positional_embedding

    def forward(self,
                target: torch.Tensor,
                mask: Optional[torch.LongTensor],
                source_hidden: torch.Tensor,
                source_mask: Optional[torch.LongTensor]
                ) -> torch.Tensor:
        """
        Transformer deocder stacked blocks.

        :param target: (batch, max_target_len, output_embedding_dim)
        :param mask: (batch, max_target_len), 0 or 1 as a mask matrix
        :param source_hidden: (batch, max_source_len, source_embedding_dim)
        :param source_mask: (batch, max_source_len)
        :return: (batch, max_target_len, output_embedding_dim)
        """
        output_tensor = target
        if self._emb_mapper is not None:
            output_tensor = self._emb_mapper(output_tensor)
        if self._use_positional_embedding:
            output_tensor = add_positional_features(output_tensor)

        for (masked_attention,
             masked_attention_norm,
             attention,
             attention_norm,
             feedforward,
             feedforward_norm) in zip(self._mask_attention_layers,
                                      self._mask_attention_norm_layers,
                                      self._attention_layers,
                                      self._attention_norm_layers,
                                      self._feedforward_layers,
                                      self._feedforward_norm_layers):

            masked_attention_out, _ = masked_attention(output_tensor, mask)
            masked_attention_out = self._dropout(masked_attention_out)           # add residual dropout
            masked_attention_out = masked_attention_norm(masked_attention_out + output_tensor)  # add residual connection

            attention_out, _ = attention(masked_attention_out, source_hidden, source_mask)
            attention_out = self._dropout(attention_out)                         # add residual dropout
            attention_out = attention_norm(attention_out + masked_attention_out) # add residual connection

            feedforward_out = feedforward(attention_out)
            feedforward_out = self._dropout(feedforward_out)                     # add residual dropout
            feedforward_out = feedforward_norm(feedforward_out + attention_out)  # add residual connection

            output_tensor = feedforward_out

        return output_tensor


class UniversalTransformerDecoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,  # input embedding dimension
                 hidden_dim: int = None,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 feedforward_hidden_dim: int = None,
                 feedforward_hidden_activation: str = "mish",
                 feedforward_dropout: float = 0.1,
                 attention_dim: int = None,
                 value_dim: int = None,
                 residual_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 ):
        """
        Construct a decoder for transformer, which is in charge of modules in the transformer model
        from the Positional Embedding before the final linear projection.
        The embedding and linear projection should be implemented elsewhere.

        :param num_layers: the number of stack layers of the transformer block
        """
        super().__init__()

        hidden_dim = hidden_dim or input_dim  # the hidden states dimension outputted by the decoder module
        self._emb_mapper = None if hidden_dim == input_dim else torch.nn.Linear(input_dim, hidden_dim)

        attention_dim = attention_dim or (hidden_dim // num_heads)
        value_dim = value_dim or (hidden_dim // num_heads)
        feedforward_hidden_dim = feedforward_hidden_dim or hidden_dim

        self.num_layers = num_layers

        self.masked_attention = MaskedMultiHeadSelfAttention(num_heads,
                                                             hidden_dim,
                                                             attention_dim * num_heads,
                                                             value_dim * num_heads,
                                                             attention_dropout=attention_dropout)

        self.masked_attention_norm = LayerNorm(hidden_dim)

        self.attention = MultiHeadAttention(num_heads,
                                           hidden_dim,
                                           hidden_dim,
                                           attention_dim * num_heads,
                                           value_dim * num_heads,
                                           attention_dropout=attention_dropout)

        self.attention_norm = LayerNorm(hidden_dim)

        self.feedfoward = FeedForward(hidden_dim,
                                     num_layers=2,
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     activations=[Activation.by_name(feedforward_hidden_activation)(),
                                                  Activation.by_name('linear')()],
                                     dropout=feedforward_dropout)

        self.feedforward_norm = LayerNorm(hidden_dim)

        self.dropout = torch.nn.Dropout(residual_dropout)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self,
                target: torch.Tensor,
                mask: Optional[torch.LongTensor],
                source_hidden: torch.Tensor,
                source_mask: Optional[torch.LongTensor]
                ) -> torch.Tensor:
        """
        Transformer deocder stacked blocks.

        :param target: (batch, max_target_len, output_embedding_dim)
        :param mask: (batch, max_target_len), 0 or 1 as a mask matrix
        :param source_hidden: (batch, max_source_len, source_embedding_dim)
        :param source_mask: (batch, max_source_len)
        :return: (batch, max_target_len, output_embedding_dim)
        """
        output_tensor = target
        if self._emb_mapper is not None:
            output_tensor = self._emb_mapper(output_tensor)

        for i in range(self.num_layers):
            output_tensor = add_depth_features_to_single_position(output_tensor, i)

            self_attention_out, _ = self.masked_attention(output_tensor, mask)
            self_attention_out = self.dropout(self_attention_out + output_tensor)
            self_attention_out = self.masked_attention_norm(self_attention_out)

            attention_out, _ = self.attention(self_attention_out, source_hidden, source_mask)
            attention_out = self.dropout(attention_out + self_attention_out)
            attention_out = self.attention_norm(attention_out)

            ffn_out = self.feedfoward(attention_out)
            ffn_out = self.dropout(ffn_out + attention_out)
            output_tensor = self.feedforward_norm(ffn_out)

        return output_tensor

