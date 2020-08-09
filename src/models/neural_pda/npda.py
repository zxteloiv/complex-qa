import torch
from torch import nn
from typing import List

class NeuralPDA(torch.nn.Module):
    def __init__(self,
                 token_embedding: nn.Embedding,
                 stack_embedding: nn.Embedding,
                 validation_classifier,

                 ):
        super().__init__()

    def forward(self):
        pass