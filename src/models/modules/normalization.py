import torch.nn.functional


class Normalization(torch.nn.Module):
    def __init__(self, p=2, dim=-1, eps=1e-2):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, input):
        return torch.nn.functional.normalize(input, self.p, self.dim, self.eps)

