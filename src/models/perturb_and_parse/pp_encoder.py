from typing import List

import torch

from models.interfaces.encoder import EncoderStack
from .eisner_surrogate import eisner_surrogate
from .gcn_layer import GCN


class PerturbParseEncoder(EncoderStack):
    def get_layer_num(self) -> int:
        return self.num_gcn_layers

    def get_layered_output(self) -> List[torch.Tensor]:
        return self._latest_output

    def forward(self, inputs, mask) -> torch.Tensor:
        """
        :param inputs: (batch, seq, inp_sz)
        :param mask: (batch, seq)
        :return: (batch, seq, hid_sz)
        """
        self._latest_output = None
        weights = self._get_arc_weights(inputs, mask)
        perturb_weights = self._perturb(weights)
        dep_tree = self._parse(perturb_weights, mask, hard=not self.training)

        output = []
        x = inputs
        for gcn in self.gcn_tower:
            gcn: GCN
            x = gcn(x, dep_tree)
            output.append(x)

        self._latest_output = output
        return output[-1]

    def _get_arc_weights(self, inputs, mask):
        # weights: (batch, seq)
        head_ = self.mlp_head(inputs)   # (batch, seq, hid_sz)
        tail_ = self.mlp_tail(inputs).transpose(1, 2)   # (batch, hid_sz, seq)
        weights = torch.bmm(head_, tail_)   # (batch, seq, seq)

        if self.distance_bias is not None:
            bias = self.distance_bias[self._build_bias_matrix(mask)]     # (batch, seq, seq)
            weights = weights + bias

        return weights

    def _build_bias_matrix(self, mask):
        batch, seq_len = mask.size()
        dev = mask.device
        dist = torch.arange(seq_len, device=dev).unsqueeze(1) - torch.arange(seq_len, device=dev).unsqueeze(0)
        dist = torch.clamp_(dist, -self.max_dep_length, self.max_dep_length - 1) + self.max_dep_length
        dist = dist.unsqueeze(0)    # (batch, seq, seq), d[i, j] = clamp(i - j) + shift
        return dist

    def _perturb(self, unnormalized_weights: torch.Tensor) -> torch.Tensor:
        dev = unnormalized_weights.device
        standard_gumbel = torch.distributions.Gumbel(torch.tensor([0.], device=dev), torch.tensor([1.], device=dev))
        noise = standard_gumbel.sample(unnormalized_weights.size())
        perturbed_weights = unnormalized_weights + noise.squeeze()
        return perturbed_weights

    def _parse(self,
               weights: torch.Tensor,
               mask: torch.Tensor = None,
               hard: bool = False) -> torch.Tensor:
        # (batch, n, n)
        dep_tree = eisner_surrogate(weights, mask, hard)
        # For now the parsing algorithm doesn't recognize paddings,
        # we resort to manually erase the nodes from dep trees instead.
        # The dep_tree is a directed arc matrix (batch, n, n), indicating
        # the existence of edges from node i to j within the batch b.
        #
        # Specifically, in our implementation of eisner_surrogate,
        #      dep_tree.sum(dim=1) = 1, out of dim0 and dim2, i.e.,
        # for each node j, the distribution over all other nodes forms
        # a probability measure, telling its parent node.

        # (batch, n, n)
        graph_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        erased_tree = dep_tree * graph_mask
        parent_norm = erased_tree.sum(1, keepdim=True)
        parent_norm += (~(parent_norm > 0)) * 1e-16
        rescaled_tree = erased_tree / parent_norm
        return rescaled_tree    # (batch, n, n)

    def is_bidirectional(self) -> bool:
        return False

    def get_input_dim(self) -> int:
        return self.inp_sz

    def get_output_dim(self) -> int:
        return self.hid_sz

    def __init__(self, inp_sz, hid_sz,
                 num_gcn_layers: int = 1,
                 max_dep_length: int = 20,
                 use_distance_bias: bool = True,
                 gcn_activation: str = 'mish',
                 ):
        super().__init__()
        self.inp_sz = inp_sz
        self.hid_sz = hid_sz
        self.num_gcn_layers = num_gcn_layers
        self.max_dep_length = max_dep_length

        self._latest_output = None
        # use the original model for raw edge weights:
        #   W_{h,m} = mlp_head(e_h)^T mlp_tail(e_m) + bias[h - m]
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(inp_sz, hid_sz),
            torch.nn.Tanh(),
            torch.nn.Linear(hid_sz, hid_sz),
            torch.nn.Tanh(),
        )
        self.mlp_tail = torch.nn.Sequential(
            torch.nn.Linear(inp_sz, hid_sz),
            torch.nn.Tanh(),
            torch.nn.Linear(hid_sz, hid_sz),
            torch.nn.Tanh(),
        )
        if use_distance_bias:
            self.distance_bias = torch.nn.Parameter(torch.zeros(max_dep_length * 2))
        else:
            self.distance_bias = None

        self.gcn_tower = torch.nn.ModuleList([
            GCN(inp_sz if layer == 0 else hid_sz, hid_sz, gcn_activation)
            for layer in range(self.num_gcn_layers)
        ])
