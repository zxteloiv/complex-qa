import torch
from torch import nn
from torch.nn import functional as F
from ..modules.variational_dropout import VariationalDropout

class TopDownTreeEncoder(nn.Module):
    def forward(self, tree_embedding, node_connection, node_mask, tree_hx=None):
        """
        :param tree_embedding: (batch, node_num, input_sz)
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz)
        """
        raise NotImplementedError

class TopDownLSTMEncoder(TopDownTreeEncoder):
    def __init__(self, input_sz: int, hidden_sz: int, transition_matrix_rank: int = 0, dropout=0.):
        super().__init__()
        self.inp_mapping_o = nn.Linear(input_sz, hidden_sz)
        self.inp_mapping_f = nn.Linear(input_sz, hidden_sz)
        self.inp_mapping_z = nn.Linear(input_sz, hidden_sz)
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
        inp_f = self.inp_mapping_f(tree_embedding)
        inp_o = self.inp_mapping_o(tree_embedding)
        inp_z = self.inp_mapping_z(tree_embedding)

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
                parent_h = self.hidden_dropout(parent_h)
                parent_c = torch.stack(tree_cs, dim=1)[batch_index, parent_node_id]
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

class BareDotProdAttnEncoder(TopDownTreeEncoder):
    def __init__(self, activation: nn.Module = None):
        super().__init__()
        self.activation = activation

    def forward(self, tree_embedding, node_connection, node_mask, tree_hx=None):
        """
        :param tree_embedding: (batch, node_num, input_sz)
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz)
        """
        batch, node_num, _ = tree_embedding.size()
        batch_index = torch.arange(batch, dtype=torch.long, device=tree_embedding.device)

        # tree_hs: [(batch, hid)]
        if tree_hx is None:
            tree_hs = []
            start_node = 0
        else:
            tree_hs, last_node_mask = tree_hx
            # in hs and cs there are padding values indicated by the node mask at previous time
            # these values are invalid and must be recomputed now.
            start_node = last_node_mask.sum(-1).min().item()

        for node_id in range(start_node, node_num):
            node_emb = tree_embedding[batch_index, node_id]
            if node_id == 0:
                h = node_emb

            else:
                parent_node_id = node_connection[:, node_id]    # parent_node_id: (batch,)
                parent_h = torch.stack(tree_hs, dim=1)[batch_index, parent_node_id]
                # alpha, beta, w_h, w_x: (batch,)
                alpha = (parent_h * node_emb).sum(dim=-1, keepdim=True).exp()
                beta = (node_emb * node_emb).sum(dim=-1, keepdim=True).exp()
                w_h = alpha / (alpha + beta + 1e-15)
                w_x = beta / (alpha + beta + 1e-15)

                # h: (batch, hid)
                h = (w_h * parent_h + w_x * node_emb)
                if self.activation is not None:
                    h = self.activation(h)

            if node_id < len(tree_hs):
                tree_hs[node_id] = h
            else:
                tree_hs.append(h)

        tree_h = torch.stack(tree_hs, dim=1)
        return tree_h, (tree_hs, node_mask)


if __name__ == '__main__':
    enc = TopDownLSTMEncoder(300, 128, 64)
    x = torch.randn(32, 1000, 300)
    conn = torch.ones(32, 1000).long().cumsum(dim=-1) - 2
    y, _ = enc.forward(x, conn, None)
    print(y.size())
    loss = y.sum()
    loss.backward()
    print(loss)



