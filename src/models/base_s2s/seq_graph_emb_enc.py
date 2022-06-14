from typing import Tuple, List

import torch.nn

from ..interfaces.encoder import EmbedAndGraphEncode
from models.modules.variational_dropout import VariationalDropout
from models.perturb_and_parse.gcn_layer import GCN


class GraphEmbedEncoder(EmbedAndGraphEncode):
    def is_bidirectional(self):
        return False

    def get_output_dim(self) -> int:
        return self.hid_sz

    def __init__(self,
                 source_embedding: torch.nn.Embedding,
                 emb_sz: int,
                 hid_sz: int,
                 num_gcn_layers: int = 1,
                 gcn_activation: str = "mish",
                 enc_dropout: float = 0.,
                 padding_id: int = 0,
                 ):
        super().__init__()
        self.src_embedding = source_embedding
        self.src_emb_dropout = VariationalDropout(enc_dropout, on_the_fly=True)
        self.padding = padding_id
        self.hid_sz = hid_sz
        self.num_gcn_layers = num_gcn_layers

        self.gcn_tower = torch.nn.ModuleList([
            GCN(emb_sz if layer == 0 else hid_sz, hid_sz, gcn_activation)
            for layer in range(self.num_gcn_layers)
        ])
        self._iter_graph = None

    def set_graph(self, graph):
        self._iter_graph = graph.float()

    def embed(self, source):
        # source: (batch, max_input_length), source sequence token ids
        # source_mask: (batch, max_input_length), source sequence padding mask
        # source_embedding: (batch, max_input_length, embedding_sz)
        source_embedding = self.src_embedding(source)
        source_embedding = self.src_emb_dropout(source_embedding)
        return source_embedding

    def forward(self, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        mask = (tokens != self.padding).long()
        layer_outputs = []
        x = self.embed(tokens)
        for gcn in self.gcn_tower:
            x = gcn(x, self._iter_graph)
            layer_outputs.append(x)

        return layer_outputs, mask
