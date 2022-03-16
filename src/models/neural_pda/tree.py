__doc__ = """
The file is a naive implementation of the tree in pytorch tensors.
The basic idea is to represent connections between the tree nodes via a vector.
Each element with index i indicates the parent node id of the node i.

During the tree generation process, the nodes would never be deleted, so it's simply implemented via batch_stack.
"""

from typing import Optional, Tuple
import torch
from models.modules.batched_stack import TensorBatchStack
_TT = torch.Tensor
_LT = torch.LongTensor
_FT = torch.FloatTensor
_Nullable = Optional

class Tree:
    def __init__(self,
                 batch_sz: int,
                 max_node_num: int,
                 node_value_dim: int = 1,
                 dtype=torch.long,
                 device: Optional[torch.device] = None,
                 ):
        self._node_registry = TensorBatchStack(batch_sz, max_node_num, node_value_dim, dtype, device)
        self._parent_index = TensorBatchStack(batch_sz, max_node_num, 1, dtype=torch.long, device=device)

        self.val_dim = node_value_dim
        self.max_node_num = max_node_num
        self.batch_sz = batch_sz

    def reset(self, batch: int, device: torch.device):
        self._node_registry.reset(batch, device)
        self._parent_index.reset(batch, device)
        self.batch_sz = batch

    def _apply_for_node(self, node_val: _TT, push_mask: _Nullable[_LT]) -> Tuple[_LT, _LT]:
        """
        Push new values onto the registry, and return its index.

        :param node_val: (batch, node_val_dim)
        :param push_mask: (batch,), 1 if the push is required, otherwise not touched.
        :return: Tuple of 2 tensors:
                (batch,) indicating the id assigned to the newly inserted value
                (batch,) whether the process is success, if push_mask is set to 0 then success is always achieved
        """
        assert node_val.size() == (self.batch_sz, self.val_dim)
        assert push_mask is None or push_mask.size() == (self.batch_sz,)
        if push_mask is None:
            push_mask = torch.ones(self.batch_sz, dtype=torch.long, device=node_val.device)

        succ = self._node_registry.push(node_val, push_mask)    # succ: (batch,)
        pos = self._node_registry._top_cur                      # pos: (batch,)
        return pos, succ

    def _add_next_node_adjacency(self, parent_idx: _LT, push_mask: _Nullable[_LT]):
        """
        Add to adjacency. Modification not supported via the interface.
        Adjacency must be add for the next succeeding idx.

        :param parent_idx: (batch,)
        :param push_mask: (batch,)
        :return: (batch,), successful or not, if push_mask is set to 0 then success is always returned
        """
        assert parent_idx.size() == (self.batch_sz,)
        assert push_mask is None or push_mask.size() == (self.batch_sz,)
        if push_mask is None:
            push_mask = torch.ones(self.batch_sz, dtype=torch.long, device=parent_idx.device)

        succ = self._parent_index.push(parent_idx.unsqueeze(-1), push_mask)
        return succ

    def init_root(self, root_val) -> None:
        """
        :param root_val: (batch, node_val_dim)
        :return:
        """
        root_pos, root_succ = self._apply_for_node(root_val, push_mask=None)
        dummy_parent_for_root = torch.zeros(self.batch_sz, device=root_val.device).long()
        self._add_next_node_adjacency(dummy_parent_for_root, push_mask=None)

    def add_new_node_edge(self, node_val: _TT, parent_idx: _LT, push_mask: _Nullable[_LT]):
        """
        Add a node and connect it to its parent.

        :param node_val: (batch, node_val_dim)
        :param parent_idx: (batch,)
        :param push_mask: (batch,)
        :return: Tuple of 2 tensors:
                (batch,) indicating the id assigned to the newly inserted value
                (batch,) whether the process is success, if push_mask is set to 0 then success is always achieved
        """
        node_idx, succ = self._apply_for_node(node_val, push_mask)
        adj_succ = self._add_next_node_adjacency(parent_idx, push_mask)
        assert (succ == adj_succ).all()
        return node_idx, succ

    def dump_partial_tree(self) -> Tuple[_TT, _LT, _LT]:
        """
        :return: Tuple of 3 tensors
                (batch, node_num, node_val_dim), node values
                (batch, node_num,)  node parent idx
                (batch, node_num,)  node mask
        """
        node_val, node_mask = self._node_registry.dump()
        parent_idx, parent_mask = self._parent_index.dump()

        # (batch, node_num)
        parent_idx = parent_idx.squeeze(-1)
        assert (parent_mask == node_mask).all()
        return node_val, parent_idx, parent_mask

if __name__ == '__main__':
    t = Tree(2, 7)
    def _print_tree(tree):
        n, p, m = tree.dump_partial_tree()
        print(n.squeeze(-1).tolist())
        print(p.tolist())
        print(m.tolist())
        print('-----' * 10)

    t.init_root(torch.tensor([[192], [192]]).long())
    _print_tree(t)
    node_idx, succ = t.add_new_node_edge(
        node_val=torch.full((2, 1), 61, dtype=torch.long).long(),
        parent_idx=torch.tensor([0, 0]).long(),
        push_mask=torch.tensor([1, 1]).long()
    )

    _print_tree(t)
    node_idx, succ = t.add_new_node_edge(
        node_val=torch.full((2, 1), 62, dtype=torch.long).long(),
        parent_idx=node_idx,
        push_mask=torch.tensor([0, 1]).long()
    )
    _print_tree(t)
    node_idx, succ = t.add_new_node_edge(
        node_val=torch.full((2, 1), 144, dtype=torch.long).long(),
        parent_idx=node_idx,
        push_mask=torch.tensor([1, 1]).long()
    )
    _print_tree(t)
    node_idx, succ = t.add_new_node_edge(
        node_val=torch.full((2, 1), 144, dtype=torch.long).long(),
        parent_idx=node_idx,
        push_mask=torch.tensor([1, 1]).long()
    )
    _print_tree(t)
