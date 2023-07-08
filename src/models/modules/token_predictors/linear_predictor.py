from models.interfaces.token_predictor import TokenPredictor, PredSemantics
import torch


class LinearPredictor(TokenPredictor):
    def __init__(self, input_size: int, output_size: int, output_as: PredSemantics = PredSemantics.logits):
        super().__init__(output_as=output_as)
        self.proj = torch.nn.Linear(input_size, output_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.output_semantic == PredSemantics.logits:
            return self.proj(hidden)
        else:
            return self.proj(hidden).softmax(dim=-1)
