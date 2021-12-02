from typing import Tuple, Any
import torch
from torch import nn
from torch.nn import functional as F
from ..modules.variational_dropout import VariationalDropout
from ..modules.decomposed_bilinear import DecomposedBilinear
import math


class TopDownTreeEncoder(nn.Module):
    def forward(self, tree_embedding, node_connection, node_mask, tree_hx=None) -> Tuple[torch.Tensor, Any]:
        """
        :param tree_embedding: (batch, node_num, input_sz)
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz)
        """
        raise NotImplementedError


class TopDownLSTMEncoder(TopDownTreeEncoder):
    def __init__(self, input_sz: int, hidden_sz: int, transition_matrix_rank: int = 0, dropout=0.,
                 parent_detach: bool = False):
        super().__init__()
        self.inp_mapping = nn.Linear(input_sz, hidden_sz * 3)
        self.input_dropout = VariationalDropout(dropout)
        self.hidden_dropout = VariationalDropout(dropout)

        if transition_matrix_rank == 0:
            transition_matrix_rank = hidden_sz

        self.hid_trans_o = nn.Parameter(torch.empty(transition_matrix_rank, hidden_sz))
        self.hid_trans_f = nn.Parameter(torch.empty(transition_matrix_rank, hidden_sz))
        self.hid_trans_z = nn.Parameter(torch.empty(transition_matrix_rank, hidden_sz))
        # the _a_ is only used with the leaky_relu nonlinearity, which is not used in any parameter
        nn.init.kaiming_uniform_(self.hid_trans_f, a=2.2360679775, nonlinearity='sigmoid')
        nn.init.kaiming_uniform_(self.hid_trans_o, a=2.2360679775, nonlinearity='sigmoid')
        nn.init.kaiming_uniform_(self.hid_trans_z, a=2.2360679775, nonlinearity='tanh')

        self.inp_sz = input_sz
        self.hid_sz = hidden_sz
        self.trans_sz = transition_matrix_rank
        self.parent_detach = parent_detach

    def forward(self, tree_embedding, node_connection, node_mask, tree_hx=None):
        """
        :param tree_embedding: (batch, node_num, input_sz)
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz), output hidden for the entire tree
        """
        batch, node_num, _ = tree_embedding.size()
        batch_index = torch.arange(batch, dtype=torch.long, device=tree_embedding.device)

        tree_embedding = self.input_dropout(tree_embedding)

        # inp_*: (batch, node_num, hidden_sz)
        inp_f, inp_o, inp_z = self.inp_mapping(tree_embedding).split(self.hid_sz, dim=-1)

        # tree_h, tree_c: [(batch, hid)]
        if tree_hx is None:
            tree_hs = []
            tree_cs = []
            start_node = 0
        else:
            tree_hs, tree_cs, last_node_mask = tree_hx
            # in hs and cs there are padding values indicated by the node mask at previous time
            # these values are invalid and must be recomputed now.
            start_node = last_node_mask.sum(-1).min().item()

        for node_id in range(start_node, node_num):
            # p_node_id: (batch,)
            parent_node_id = node_connection[:, node_id]

            if node_id > 0:
                parent_h = torch.stack(tree_hs, dim=1)[batch_index, parent_node_id]
                parent_c = torch.stack(tree_cs, dim=1)[batch_index, parent_node_id]
                if self.parent_detach:
                    parent_h = parent_h.detach()
                    parent_c = parent_c.detach()

                parent_h = self.hidden_dropout(parent_h)
                parent_c = self.hidden_dropout(parent_c)
                parent = (parent_h, parent_c)
            else:
                parent = None

            h, c = self._run_cell(inp_f[:, node_id], inp_o[:, node_id], inp_z[:, node_id], parent)

            if node_id < len(tree_hs):
                tree_hs[node_id] = h
                tree_cs[node_id] = c
            else:
                tree_hs.append(h)
                tree_cs.append(c)

        tree_h = torch.stack(tree_hs, dim=1)
        return tree_h, (tree_hs, tree_cs, node_mask)

    def _run_cell(self, f_, o_, z_, parent=None):
        # (batch, hid)
        if parent is not None:
            parent_h, parent_c = parent
        else:
            parent_h = parent_c = None

        if parent_h is not None:
            f_ = torch.matmul(parent_h, self.v_f) + f_
            o_ = torch.matmul(parent_h, self.v_o) + o_
            z_ = torch.matmul(parent_h, self.v_z) + z_

        # f, o, z: (batch, hid)
        f = torch.sigmoid(f_)
        o = torch.sigmoid(o_)
        z = torch.tanh(z_)

        # c: (batch, hid)
        if parent_c is not None:
            c = parent_c * f + z * (1 - f)
        else:
            c = z * (1 - f)

        h = o * torch.tanh(c)
        return h, c

    @property
    def v_f(self):
        """:return: (hid, hid)"""
        return torch.matmul(self.hid_trans_f.t(), self.hid_trans_f)

    @property
    def v_o(self):
        """:return: (hid, hid)"""
        return torch.matmul(self.hid_trans_o.t(), self.hid_trans_o)

    @property
    def v_z(self):
        """:return: (hid, hid)"""
        return torch.matmul(self.hid_trans_z.t(), self.hid_trans_z)


class TopDownBilinearLSTMEncoder(TopDownTreeEncoder):
    def __init__(self,
                 input_sz: int,
                 hidden_sz: int,
                 bilinear_rank: int,
                 bilinear_pool: int,
                 use_linear: bool = False,
                 use_bias: bool = False,
                 dropout=0.,
                 parent_detach: bool = False,
                 ):
        super().__init__()
        self.o_gate = DecomposedBilinear(input_sz, hidden_sz, hidden_sz, bilinear_rank, bilinear_pool,
                                         use_linear=use_linear, use_bias=use_bias,)
        self.f_gate = DecomposedBilinear(input_sz, hidden_sz, hidden_sz, bilinear_rank, bilinear_pool,
                                         use_linear=use_linear, use_bias=use_bias,)
        self.z = DecomposedBilinear(input_sz, hidden_sz, hidden_sz, bilinear_rank, bilinear_pool,
                                    use_linear = use_linear, use_bias = use_bias,)
        self.input_dropout = VariationalDropout(dropout)
        self.hidden_dropout = VariationalDropout(dropout)
        self.parent_detach = parent_detach

    def forward(self, tree_embedding, node_connection, node_mask, tree_hx=None):
        """
        :param tree_embedding: (batch, node_num, input_sz)
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz)
        """
        batch, node_num, hid_sz = tree_embedding.size()
        batch_index = torch.arange(batch, dtype=torch.long, device=tree_embedding.device)
        tree_embedding = self.input_dropout(tree_embedding)

        # tree_hs: [(batch, hid)]
        if tree_hx is None:
            tree_hs, tree_cs, start_node = [], [], 0
        else:
            tree_hs, tree_cs, last_node_mask = tree_hx
            # in hs and cs there are padding values indicated by the node mask at previous time
            # these values are invalid and must be recomputed now.
            start_node = last_node_mask.sum(-1).min().item()

        for node_id in range(start_node, node_num):
            node_emb = tree_embedding[batch_index, node_id]
            if node_id == 0:
                parent_c = 0
                parent_h = None

            else:
                parent_node_id = node_connection[:, node_id]    # parent_node_id: (batch,)
                parent_h = torch.stack(tree_hs, dim=1)[batch_index, parent_node_id]
                parent_c = torch.stack(tree_cs, dim=1)[batch_index, parent_node_id]
                if self.parent_detach:
                    parent_h = parent_h.detach()
                    parent_c = parent_c.detach()

                parent_h = self.hidden_dropout(parent_h)
                parent_c = self.hidden_dropout(parent_c)

            f = self.f_gate(node_emb, parent_h).sigmoid()
            o = self.o_gate(node_emb, parent_h).sigmoid()
            z = self.z(node_emb, parent_h).tanh()

            c = parent_c * f + (1 - f) * z
            h = o * c.tanh()

            if node_id < len(tree_hs):
                tree_cs[node_id] = c
                tree_hs[node_id] = h
            else:
                tree_hs.append(h)
                tree_cs.append(c)

        tree_h = torch.stack(tree_hs, dim=1)
        return tree_h, (tree_hs, tree_cs, node_mask)


class SingleStepTreeEncoder(nn.Module):
    def forward(self, tree_embedding, node_connection, node_mask) -> torch.Tensor:
        """
        :param tree_embedding: (batch, node_num, input_sz)
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz)
        """
        raise NotImplementedError


class ReZeroEncoder(TopDownTreeEncoder):
    def __init__(self, num_layers: int, layer_encoder: SingleStepTreeEncoder, activation=None):
        super().__init__()
        self._layer_enc = layer_encoder
        self.num_layers = num_layers
        self.alpha = nn.Parameter(torch.zeros(num_layers))
        self.activation = activation

    def forward(self, tree_embedding, node_connection, node_mask, tree_hx=None) -> Tuple[torch.Tensor, Any]:
        """
        :param tree_embedding: (batch, node_num, hidden_sz), requires input_size == output_size
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz)
        """
        layer_hid = tree_embedding
        for depth in range(self.num_layers):
            new_hid = self._layer_enc(layer_hid, node_connection, node_mask)
            if self.activation is not None:
                new_hid = self.activation(new_hid)
            layer_hid = layer_hid + new_hid * self.alpha[depth]

        return layer_hid, None


class SingleStepBilinear(SingleStepTreeEncoder):
    def __init__(self, mod: DecomposedBilinear):
        super().__init__()
        self.mod = mod

    def forward(self, tree_embedding, node_connection, node_mask) -> torch.Tensor:
        batch, node_num, hid_sz = tree_embedding.size()
        batch_index = torch.arange(batch, dtype=torch.long, device=tree_embedding.device)
        parent_hid = tree_embedding[batch_index.unsqueeze(-1), node_connection]
        return self.mod(tree_embedding, parent_hid)


class SingleStepDotProd(SingleStepTreeEncoder):
    def forward(self, tree_embedding, node_connection, node_mask) -> torch.Tensor:
        """
        :param tree_embedding: (batch, node_num, hidden_sz), requires input_size == output_size
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz)
        """
        batch, node_num, hid_sz = tree_embedding.size()
        batch_index = torch.arange(batch, dtype=torch.long, device=tree_embedding.device)
        # parent_h: (batch, node, hid)
        parent_h = tree_embedding[batch_index.unsqueeze(-1), node_connection]
        # alpha, beta, w_h, w_x: (batch, node, 1)
        alpha = ((parent_h * tree_embedding).sum(dim=-1, keepdim=True) / math.sqrt(hid_sz)).exp()
        beta = ((tree_embedding * tree_embedding).sum(dim=-1, keepdim=True) / math.sqrt(hid_sz)).exp()
        w_h = (alpha + 1e-15) / (alpha + beta + 1e-15)
        w_x = beta / (alpha + beta + 1e-15)

        # h: (batch, hid)
        h = (w_h * parent_h + w_x * tree_embedding)
        return h


if __name__ == '__main__':
    enc = TopDownLSTMEncoder(300, 128, 64)
    x = torch.randn(32, 1000, 300)
    conn = torch.ones(32, 1000).long().cumsum(dim=-1) - 2
    y, _ = enc.forward(x, conn, None)
    print(y.size())
    loss = y.sum()
    loss.backward()
    print(loss)



