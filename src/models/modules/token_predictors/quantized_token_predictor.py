import torch
from torch import nn
from models.interfaces.token_predictor import TokenPredictor, PredSemantics
from enum import auto, StrEnum


class QuantMethod(StrEnum):
    """
    If we relate the hidden state $h$ with the embedding matrix $E$, and take the process as quantization,
    there are a few methods that can implement the quantization.

    - by distance: Negative euclidean distances of h and every e are used for logits.
                    The closer embedding indicates larger logit.
    - by dot_product: working like a linear layer without bias,
                    where greater dot-product yields larger logit;
    - by projection: the h is projected onto each e, the projection is used as logit.
                    where greater projection yields larger logit;
    - by cosine: logit is the cosine of theta between the h and each e,
               where smaller angle has larger cosine and logit;
    """
    distance = auto()
    projection = auto()
    dot_product = auto()
    cosine = auto()


class QuantTokenPredictor(TokenPredictor):
    def __init__(self,
                 emb_sz: int,
                 num_toks: int,
                 shared_embedding: nn.Parameter | None = None,
                 quant_criterion: QuantMethod = QuantMethod.distance,
                 output_as: PredSemantics = PredSemantics.logits,
                 ):
        """
        :param emb_sz: int, the token embedding size,
        :param num_toks: int, the number of tokens
        :param shared_embedding: if given, must be of (num_tokens, embedding_size)
        :param quant_criterion: see the QuantMethod doc for explanation.
        """
        super().__init__(output_as=output_as)

        if shared_embedding is None:
            weight = torch.Tensor(num_toks, emb_sz)
            nn.init.normal_(weight)
            nn.functional.normalize(weight, dim=-1, out=weight)
            self.weight = nn.Parameter(weight)

        else:
            assert shared_embedding.size() == (num_toks, emb_sz)
            self.weight = shared_embedding

        self.quant_criterion = quant_criterion
        self.num_toks = num_toks
        self.emb_sz = emb_sz

    def forward(self, h):
        """
        :param h: (..., hidden_dim)
        :return: (..., num_toks)
        """
        assert h.size(-1) == self.emb_sz, "Quantization can only be applied on the same embedding size"

        if self.quant_criterion == QuantMethod.distance:
            # h_rs: (..., 1, emb_sz)
            h_rs = h.unsqueeze(-2)
            # weight: (num_toks, emb_sz)
            # dist: (..., num_toks)
            dist = (h_rs - self.weight).norm(dim=-1)
            output = -dist
        else:
            # h: (..., emb_sz)
            # weight: (num_toks, emb_sz)
            # dot_prod, proj: (..., num_toks)
            dot_prod = torch.matmul(h, self.weight.t())
            output = dot_prod
            if self.quant_criterion == QuantMethod.projection:
                proj = dot_prod / self.weight.norm(dim=-1)
                output = proj
            elif self.quant_criterion == QuantMethod.cosine:
                proj = dot_prod / self.weight.norm(dim=-1) / h.norm(dim=-1, keepdim=True)
                output = proj

        if self.output_semantic == PredSemantics.probs:
            output = output.softmax(dim=-1)
        return output
