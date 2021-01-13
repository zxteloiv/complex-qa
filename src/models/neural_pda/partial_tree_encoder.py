import torch
from torch import nn
from torch.nn import functional as F

class TopDownTreeEncoder(nn.Module):
    def forward(self, tree_embedding, node_connection, node_mask):
        """
        :param tree_embedding: (batch, node_num, input_sz)
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz)
        """
        raise NotImplementedError

class TopDownLSTMEncoder(TopDownTreeEncoder):
    def __init__(self, input_sz: int, hidden_sz: int, transition_matrix_rank: int = 0, tensor_based: bool = True):
        super().__init__()
        self.inp_mapping_o = nn.Linear(input_sz, hidden_sz)
        self.inp_mapping_f = nn.Linear(input_sz, hidden_sz)
        self.inp_mapping_z = nn.Linear(input_sz, hidden_sz)

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

        self.tensor_based = tensor_based

    def forward(self, tree_embedding, node_connection, node_mask):
        """
        :param tree_embedding: (batch, node_num, input_sz)
        :param node_connection: (batch, node_num)
        :param node_mask: (batch, node_num)
        :return: (batch, node_num, hidden_sz), output hidden for the entire tree
        """
        batch, node_num, _ = tree_embedding.size()
        batch_index = torch.arange(batch, dtype=torch.long, device=tree_embedding.device)

        # inp_*: (batch, node_num, hidden_sz)
        inp_f = self.inp_mapping_f(tree_embedding)
        inp_o = self.inp_mapping_o(tree_embedding)
        inp_z = self.inp_mapping_z(tree_embedding)

        # tree_h, tree_c: [(batch, hid)]
        if self.tensor_based:
            tree_h = torch.randn_like(inp_f)
            tree_c = torch.randn_like(inp_f)
        else:
            tree_hs = []
            tree_cs = []

        for node_id in range(node_num):
            # p_node_id: (batch,)
            parent_node_id = node_connection[:, node_id]

            if node_id > 0:
                if self.tensor_based:
                    parent_h = tree_h[batch_index, parent_node_id]
                    parent_c = tree_c[batch_index, parent_node_id]
                else:
                    parent_h = torch.stack(tree_hs, dim=1)[batch_index, parent_node_id]
                    parent_c = torch.stack(tree_cs, dim=1)[batch_index, parent_node_id]
                parent = (parent_h, parent_c)
            else:
                parent = None

            h, c = self._run_cell(inp_f, inp_o, inp_z, parent)

            if self.tensor_based:
                tree_h[batch_index, node_id] = h
                tree_c[batch_index, node_id] = c
            else:
                tree_hs.append(h)
                tree_cs.append(c)

        if not self.tensor_based:
            tree_h = torch.stack(tree_hs, dim=1)
        return tree_h

    def _run_cell(self, inp_f, inp_o, inp_z, parent=None):
        # (batch, hid)
        f_, o_, z_ = list(map(lambda t: t[:, 0, :], (inp_f, inp_o, inp_z)))
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


if __name__ == '__main__':
    enc = TopDownLSTMEncoder(300, 128, 64)
    x = torch.randn(32, 1000, 300)
    conn = torch.ones(32, 1000).long().cumsum(dim=-1) - 2
    y = enc.forward(x, conn, None)
    print(y.size())
    loss = y.sum()
    loss.backward()
    print(loss)



