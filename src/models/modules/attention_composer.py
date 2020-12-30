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
    def __init__(self, context_dim, vector_dim, output_dim, dropout=.0):
        super().__init__()

        self.context_maping = torch.nn.Linear(context_dim, output_dim, bias=False)
        self.vector_mapping = torch.nn.Linear(vector_dim, output_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dropout(self.context_maping(context) + self.vector_mapping(hidden)))

