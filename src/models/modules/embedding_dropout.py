import torch


class SeqEmbeddingDropoutWrapper:
    """
    Add several dropout to the embedding tensor directly after the word embedding module.
    """
    def __init__(self,
                 emb: torch.nn.Embedding,
                 discrete_dropout: float = 0.,
                 input_dropout: float = 0.,
                 ):
        super().__init__()
        self.d_dropout = torch.nn.Dropout2d(discrete_dropout)
        self.i_dropout = torch.nn.Dropout(input_dropout)
        self.emb = emb

    def forward(self, tok: torch.LongTensor) -> torch.Tensor:
        """
        Applied the dropout for the
        :param tok: the input tokens tensor, usually in shape (batch, length)
        :return: tensor with multiple dropout applied, in shape (batch, length, embedding)
        """
        embedding = self.emb(tok)
        embedding = self.d_dropout(embedding.unsqueeze(-1)).squeeze(-1)
        embedding = self.i_dropout(embedding)

        return embedding

