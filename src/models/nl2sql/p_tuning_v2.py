# implement a similar P-Tuning V2 module from the original implementations
import torch
from torch import nn

from models.interfaces.encoder import Encoder


class PTuningV2Prompt(nn.Module):
    def __init__(self,
                 prefix_len: int,
                 n_layers: int,
                 n_head: int,
                 plm_hidden: int,
                 prefix_enc: Encoder = None,
                 dropout: nn.Module = None,
                 ):
        super().__init__()
        self.prefix_len = prefix_len
        if prefix_enc is None:
            self.emb = nn.Embedding(prefix_len, plm_hidden * n_layers * 2)
        else:
            self.emb = nn.Embedding(prefix_len, prefix_enc.get_input_dim())
            assert prefix_enc.get_output_dim() == plm_hidden * n_layers * 2

        self.prefix_encoder = prefix_enc
        self.dropout = dropout

        self.n_layers = n_layers
        self.n_head = n_head
        self.plm_hidden = plm_hidden
        self.head_emb_sz = plm_hidden // n_head

    def forward(self, nbatch):
        dev = self.emb.weight.device
        batch_tokens = torch.arange(self.prefix_len, device=dev, dtype=torch.long).unsqueeze(0).expand(nbatch, -1)
        embeddings = self.emb(batch_tokens)     # (b, prefix_len, plm_hidden * #layers * 2)
        if self.prefix_encoder is not None:
            embeddings = self.prefix_encoder(embeddings)

        past_key_values = embeddings.view(
            nbatch, self.prefix_len, self.n_layers * 2, self.n_head, self.head_emb_sz
        )
        if self.dropout:
            past_key_values = self.dropout(past_key_values)

        # the bert model requires individual dimensions as
        #      list     1D-tensor Len2                4D-tensor
        #    (layers,   (key and val),     (batch, n_head, prefix_len, head_emb))
        # past_key_values: [(2, batch, n_head, len, head_emb)] * n_layer
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2, dim=0)

        return past_key_values
