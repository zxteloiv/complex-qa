import torch
from torch import nn
from typing import List

class NeuralPDA(torch.nn.Module):
    def __init__(self,
                 token_embedding: nn.Embedding,
                 core_decoder: nn.Module,
                 nt_decoder: nn.Module,
                 num_nonterminals: int,
                 terminal_predictor: nn.Module,
                 num_terminal: int,
                 nonterminal_dim: int,
                 ):
        super().__init__()
        self.token_embedding = token_embedding
        self.core_decoder = core_decoder
        self.nt_decoder = nt_decoder
        self.num_nt = num_nonterminals
        self.terminal_predictor = terminal_predictor
        self.num_terminal = num_terminal

        codebook = torch.randn(num_nonterminals, nonterminal_dim)
        self.codebook = nn.Parameter(codebook)

    def forward(self):
        pass