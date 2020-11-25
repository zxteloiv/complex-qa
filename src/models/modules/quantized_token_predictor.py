from typing import Union, Literal, Optional
import torch
from torch import nn

PREDICTOR_OUTPUT_SEMANTIC = Literal["probs", "logits"]

class QuantTokenPredictor(nn.Module):
    def __init__(self,
                 num_toks,
                 tok_dim,
                 output_semantics: PREDICTOR_OUTPUT_SEMANTIC = "logits",
                 shared_embedding: Optional[nn.Parameter] = None,
                 quant_criterion: Literal["distance", "projection"] = "distance",
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

        # h_rs: (..., 1, tok_dim)
        h_rs = h.unsqueeze(-2)
        if self.quant_criterion == 'distance':
            # weight: (num_toks, tok_dim)
            # dist: (..., num_toks)
            dist = (h_rs - self.weight).norm(dim=-1)
            output = dist
        else:
            # weight: (num_toks, tok_dim)
            # proj: (..., num_toks)
            proj = (h_rs * self.weight).sum(-1) / self.weight.norm(dim=-1)
            output = proj

        if self.output_probs:
            output = dist.softmax(dim=-1)
        return output
