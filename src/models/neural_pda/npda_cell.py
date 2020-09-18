from typing import Tuple, Optional, List, NamedTuple
import torch
from torch import nn

class NPDAHidden(NamedTuple):
    token: torch.Tensor
    stack: torch.Tensor
    state: torch.Tensor

class RNNPDACell(nn.Module):
    def __init__(self,
                 token_dim,
                 stack_dim,
                 hidden_dim,
                 ):
        super().__init__()

        self.token_weights = nn.Linear(token_dim, hidden_dim * 4, bias=True)
        self.stack_weights = nn.Linear(stack_dim, hidden_dim * 4, bias=False)
        self.token_dim = token_dim
        self.stack_dim = stack_dim
        self.hidden_dim = hidden_dim

    def forward(self,
                token: torch.Tensor,
                stack: torch.Tensor,
                hiddens: Optional[NPDAHidden] = None,
                ) -> NPDAHidden:
        """
        :param token: (batch, token_dim), input token of the current timestep
        :param stack: (batch, stack_dim), top item on the stack at the current time
        :param hiddens: Tuple[(batch, hidden_dim)] * 3, hidden states for terminals, non-terminals, and states
        :return: hidden state of terminals, non-terminals, and states
        """
        total: torch.Tensor = self.token_weights(token) + self.stack_weights(stack)

        # z_*: (batch, hidden_dim)
        z_t, z_nt, z_s, z_f = total.split(self.hidden_dim, dim=-1)

        if hiddens is None:
            return NPDAHidden(z_t, z_nt, z_s)

        # forget, remember: (batch, hidden_dim)
        forget = torch.sigmoid(z_f)
        remember = 1 - forget

        h_t, h_nt, h_s = hiddens.token, hiddens.stack, hiddens.state

        h_t = forget * h_t + remember * z_t
        h_nt = forget * h_nt + remember * z_nt
        h_s = forget * h_s + remember * z_s

        return NPDAHidden(h_t, h_nt, h_s)
