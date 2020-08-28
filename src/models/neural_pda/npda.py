from typing import Optional, Callable
import torch
from torch import nn
from .npda_cell import RNNPDACell

class NeuralPDA(torch.nn.Module):
    def __init__(self,
                 pda_decoder: nn.Module,
                 nt_decoder: nn.Module,
                 num_nonterminals: int,
                 num_terminal: int,
                 nonterminal_dim: int,
                 token_embedding: nn.Embedding,
                 token_predictor: nn.Module,
                 ):
        super().__init__()
        self.pda_decoder = pda_decoder
        self.nt_decoder = nt_decoder

        self.num_terminal = num_terminal
        self.num_nt = num_nonterminals
        codebook = torch.randn(num_nonterminals, nonterminal_dim)
        self.codebook = nn.Parameter(codebook)

        self.token_embedding = token_embedding
        self.token_predictor = token_predictor

        self._stack = []

    def forward(self,
                x: torch.Tensor,
                h: Optional[RNNPDACell.hidden_type] = None,
                attn_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """

        :param x: (batch, length)
        :param h: 
        :param attn_fn: a function wrapper to dynamically call to attend over other data source.
        :return:
        """
        batch, decoding_length = x.size()[0:2]
        stack_bottom = x.new_zeros((batch, self.nonterminal_dim))
        self._stack = [stack_bottom]

        hidden_by_step = []
        for step in range(decoding_length):
            # step_input: (batch, hidden_dim)
            step_tok = x[:, step]
            step_input = self.token_embedding(step_tok)





        pass