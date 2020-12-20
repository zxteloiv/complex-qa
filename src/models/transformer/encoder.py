from typing import Dict, List, Tuple, Mapping, Optional
import torch
import torch.nn

from allennlp.modules import FeedForward
from torch.nn import LayerNorm
from utils.nn import add_positional_features
from allennlp.nn import Activation

from .multi_head_attention import MultiHeadSelfAttention

class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,  # input embedding dimension
                 hidden_dim: int = None,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 feedforward_hidden_dim: int = None,
                 feedforward_dropout: float = 0.1,
                 residual_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_positional_embedding: bool = True,
                 ):
        super(TransformerEncoder, self).__init__()

        self._attention_layers: List[MultiHeadSelfAttention] = []
        self._attention_norm_layers: List[LayerNorm] = []
        self._feedforward_layers: List[FeedForward] = []
        self._feedforward_norm_layers: List[LayerNorm] = []

        hidden_dim = hidden_dim or input_dim
        feedforward_hidden_dim = feedforward_hidden_dim or hidden_dim

        layer_inp = input_dim
        for i in range(num_layers):
            attention = MultiHeadSelfAttention(num_heads, layer_inp, layer_inp, layer_inp,
                                               attention_dropout=attention_dropout)
            self.add_module(f'attention_{i}', attention)
            self._attention_layers.append(attention)

            attention_norm = LayerNorm(layer_inp)
            self.add_module(f'attention_norm_{i}', attention_norm)
            self._attention_norm_layers.append(attention_norm)

            feedfoward = FeedForward(layer_inp,
                                     num_layers=2,
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     activations=[Activation.by_name('relu')(),
                                                  Activation.by_name('linear')()],
                                     dropout=feedforward_dropout)
            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedforward_layers.append(feedfoward)

            feedforward_norm = LayerNorm(hidden_dim)
            self.add_module(f"feedforward_norm_{i}", feedforward_norm)
            self._feedforward_norm_layers.append(feedforward_norm)
            layer_inp = hidden_dim

        self._dropout = torch.nn.Dropout(residual_dropout)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._use_positional_embedding = use_positional_embedding

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output_tensor = add_positional_features(inputs) if self._use_positional_embedding else inputs

        for (attention,
             attention_norm,
             feedforward,
             feedforward_norm) in zip(self._attention_layers,
                                      self._attention_norm_layers,
                                      self._feedforward_layers,
                                      self._feedforward_norm_layers):
            cached_input = output_tensor

            attention_out, _ = attention(output_tensor, mask)
            attention_out = self._dropout(attention_out)
            attention_out = attention_norm(attention_out + cached_input)

            feedforward_out = feedforward(attention_out)
            feedforward_out = self._dropout(feedforward_out)
            feedforward_out = feedforward_norm(feedforward_out + attention_out)

            output_tensor = feedforward_out

        return output_tensor

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def is_bidirectional(self) -> bool:
        return False

    def get_input_dim(self) -> int:
        return self.input_dim


