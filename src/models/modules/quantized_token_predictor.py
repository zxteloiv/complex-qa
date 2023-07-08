from typing import Literal
import torch
from torch import nn


PREDICTOR_OUTPUT_SEMANTIC = Literal["probs", "logits"]


class QuantTokenPredictor(nn.Module):
    def __init__(self,
                 num_toks,
                 tok_dim,
                 output_semantics: PREDICTOR_OUTPUT_SEMANTIC = "logits",
                 shared_embedding: nn.Parameter | None = None,
                 quant_criterion: Literal["distance", "projection", "dot_product"] = "distance",
                 ):
        super().__init__()

        if shared_embedding is None:
            weight = torch.Tensor(num_toks, tok_dim)
            nn.init.normal_(weight)
            nn.functional.normalize(weight, dim=-1, out=weight)
            self.weight = nn.Parameter(weight)

        else:
            self.weight = shared_embedding

        self.output_probs = output_semantics == "probs"
        self.quant_criterion = quant_criterion
        self.num_toks = num_toks
        self.tok_dim = tok_dim

    def forward(self, h):
        """
        :param h: (..., hidden_dim)
        :return: (..., num_toks)
        """
        assert h.size(-1) == self.tok_dim, "Quantization can only be applied on the same embedding size"

        if self.quant_criterion == 'distance':
            # h_rs: (..., 1, tok_dim)
            h_rs = h.unsqueeze(-2)
            # weight: (num_toks, tok_dim)
            # dist: (..., num_toks)
            dist = (h_rs - self.weight).norm(dim=-1)
            output = -dist
        else:
            # h: (..., tok_dim)
            # weight: (num_toks, tok_dim)
            # dot_prod, proj: (..., num_toks)
            dot_prod = torch.matmul(h, self.weight.t())
            output = dot_prod
            if self.quant_criterion == "projection":
                proj = dot_prod / self.weight.norm(dim=-1)
                output = proj

        if self.output_probs:
            output = output.softmax(dim=-1)
        return output
