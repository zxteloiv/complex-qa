import torch

from ..interfaces.attention import VectorContextComposer


class CatComposer(VectorContextComposer):
    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([context, hidden], dim=-1)


class AddComposer(VectorContextComposer):
    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + context


class ClassicMLPComposer(VectorContextComposer):
    """
    Similar to the Attention module used by Dong and Lapata (2016).
    """
    def __init__(self, context_dim, vector_dim, output_dim, dropout=.0, use_tanh=True):
        super().__init__()

        self.context_maping = torch.nn.Linear(context_dim, output_dim, bias=False)
        self.vector_mapping = torch.nn.Linear(vector_dim, output_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.use_tanh = use_tanh

    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        z = self.context_maping(context) + self.vector_mapping(hidden)
        if self.use_tanh:
            z = torch.tanh(self.dropout(z))

        return z


