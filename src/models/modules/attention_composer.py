import torch

from ..interfaces.attention import VectorContextComposer

class CatComposer(VectorContextComposer):
    def __init__(self, context_dim: int, vector_dim: int):
        super(CatComposer, self).__init__()
        self.output_dim = context_dim + vector_dim

    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([context, hidden], dim=-1)

    def get_output_dim(self) -> int:
        return self.output_dim

class CatMappingComposer(VectorContextComposer):
    def __init__(self, context_dim: int, vector_dim: int, output_dim: int, activation: str = 'tanh'):
        super().__init__()
        self.output_dim = output_dim
        self.mapping = torch.nn.Linear(context_dim + vector_dim, output_dim)

        from allennlp.nn.activations import Activation
        self.activation = Activation.by_name(activation)()

    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        state = torch.cat([context, hidden], dim=-1)
        return self.activation(self.mapping(state))

    def get_output_dim(self) -> int:
        return self.output_dim

class AddComposer(VectorContextComposer):
    def __init__(self, both_in_out_dim: int):
        super().__init__()
        self.output_dim = both_in_out_dim

    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + context

    def get_output_dim(self) -> int:
        return self.output_dim

class MappingAddComposer(VectorContextComposer):
    """
    Similar to the Attention module used by Dong and Lapata (2016).
    """
    def __init__(self, context_dim, vector_dim, output_dim: int, activation: str = "tanh"):
        super().__init__()

        self.context_maping = torch.nn.Linear(context_dim, output_dim, bias=False)
        self.vector_mapping = torch.nn.Linear(vector_dim, output_dim, bias=False)
        self.output_dim: int = output_dim
        from allennlp.nn.activations import Activation
        self.activation = Activation.by_name(activation)()

    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        z = self.context_maping(context) + self.vector_mapping(hidden)
        z = self.activation(z)
        return z

    def get_output_dim(self) -> int:
        return self.output_dim

cls_mappings = {
    "cat": CatComposer,
    "add": AddComposer,
    "mapping_add": MappingAddComposer,
    "cat_mapping": CatMappingComposer,
}

def get_attn_composer(cls_type: str, context_dim: int, vector_dim: int, output_dim: int, activation: str):
    if cls_type == "none": return None
    elif cls_type == "cat":
        assert output_dim == context_dim + vector_dim, "CatComposer will concatenate the context and vector"
        return CatComposer(context_dim, vector_dim)
    elif cls_type == "cat_mapping":
        return CatMappingComposer(context_dim, vector_dim, output_dim, activation)
    elif cls_type == "add":
        assert context_dim == vector_dim == output_dim, "AddComposer requires all dimensions equal to sum"
        return AddComposer(context_dim)
    elif cls_type == "mapping_add":
        return MappingAddComposer(context_dim, vector_dim, output_dim, activation)
    else:
        raise ValueError(f"The {cls_type} composer is not supported. Current composers {cls_mappings.keys()}")