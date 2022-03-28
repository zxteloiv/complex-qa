from typing import Optional, Tuple, Union
import torch


class DumpBatchStack:
    def dump(self):
        raise NotImplementedError


class BatchStack:
    def reset(self, batch_size, default_device = None):
        raise NotImplementedError

    def describe(self):
        raise NotImplementedError

    def push(self, data: torch.Tensor, push_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        :param data: (batch, *) item to push onto the stack
        :param push_mask: (batch,) indicators to push items
                (1) onto the stack or not to (0) for each item in the batch
        :return: the success indicators for the push action,
                (1) if successful (either pushed or retained), (0) if max_stack_size is exceeded.
        """
        raise NotImplementedError

    def pop(self, pop_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remove the top item from each stack, and return them.
        :param pop_mask: (batch,) pop the specified item, or all the batch items if None
        :return: A tuple of the two
                (batch, *) tuple of data items popped out,
                (batch,) an errcode for each item, (1) for successful pop, (0) if the appropriate stack is empty.
        """
        raise NotImplementedError

    def top(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top item from each stack.
        :param pop_data: not only get the top item of each stack, but also remove them from the stack top.
        :param batch_size: the required batch to pop
        :param default_item: the item to fill if the stack is empty.
        :return: A tuple of the two
                (batch, *) tuple of data items at the top, which will not be popped out if pop_data is set to False.
                (batch,) an errcode for each item, (1) for successful pop, (0) if the appropriate stack is empty.
        """
        raise NotImplementedError


class TensorBatchStack(BatchStack, DumpBatchStack):

    def __init__(self,
                 max_batch_size: int,
                 max_stack_size: int = 0,
                 item_size: int = 1,
                 dtype=torch.float,
                 device=None,
                 ):
        self.max_batch_size = max_batch_size
        self.max_stack_size = max_stack_size
        self.item_size = item_size
        self._storage: Optional[torch.Tensor] = None
        self._top_cur: Optional[torch.Tensor] = None
        self.inplace = False
        self.dtype = dtype
        self.reset(max_batch_size, device)

    def describe(self):
        stat = {
            "stack_shape": (self.max_batch_size, self.max_stack_size, self.item_size),
            "stack_load": None if self._top_cur is None else self._top_cur.tolist(),
        }
        return stat

    def reset(self, batch_size, default_device: Optional[torch.device] = None):
        self._storage = torch.zeros(batch_size, self.max_stack_size, self.item_size, device=default_device, dtype=self.dtype)
        self._top_cur = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=default_device)
        self.max_batch_size = batch_size
        return self

    def push(self, data: torch.Tensor, push_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        :param data: (batch, *) item to push onto the stack
        :param push_mask: (batch,)
                indicators to push items (1) onto the stack or not to (0) for each item in the batch
        :return: the success indicators for the push action,
                (1) if successful (either pushed or retained), (0) if max_stack_size is exceeded.
        """
        batch_sz = data.size()[0]
        assert data.dtype == self._storage.dtype
        assert push_mask is None or push_mask.size()[0] == batch_sz
        push_mask = torch.tensor([1], device=data.device).long() if push_mask is None else push_mask.long()
        # force the incoming batch size equal to the storage,
        # such that no slicing is required and the implementation is easier.
        assert batch_sz == self.max_batch_size

        # stack error is only triggered when the push action is required but the stack is full.
        # stack_error: (batch,)
        stack_error = (self._top_cur >= (self.max_stack_size - 1)) * push_mask
        succ = 1 - stack_error.long()

        batch_range = torch.arange(batch_sz, device=data.device)
        # increasing the stack top cursor only if the push action is required and valid
        self._top_cur = self._top_cur + push_mask.long() * succ.long()
        val_backup = self._storage[batch_range, self._top_cur]
        push_mask = push_mask.unsqueeze(-1)
        val_new = val_backup * push_mask.logical_not() + data * push_mask

        new_storage = self._storage.clone()
        new_storage[batch_range, self._top_cur] = val_new
        self._storage = new_storage
        return succ

    def top(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top item from each stack.
        :param batch_size: the required batch to pop, must be the max_batch_size for a tensor-based BatchStack.
        :return: A tuple of the two
                (batch, *) tuple of data items popped out,
                (batch,) an errcode for each item, (1) for successful pop, (0) if the appropriate stack is empty.
        """
        batch_size, device = self.max_batch_size, self._storage.device

        # top_succ, top_idx: (batch,)
        top_succ = ((0 <= self._top_cur) * (self._top_cur < self.max_stack_size)).long()
        top_idx = self._top_cur * top_succ  # invalid position has the default index 0

        # batch_range: (batch,)
        # val_mask: (batch, 1)
        batch_range = torch.arange(batch_size, device=device)
        val_mask = top_succ.unsqueeze(-1)
        top_val = self._storage[batch_range, top_idx] * val_mask    # any invalid or unwanted item has value 0

        return top_val, top_succ

    def pop(self, pop_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # top_val: (batch, *)
        # top_succ: (batch,)
        top_val, top_succ = self.top()

        if pop_mask is None:    # pop all the data if possible
            self._top_cur = self._top_cur - top_succ
            return top_val, top_succ

        assert pop_mask.size() == (self.max_batch_size,)
        # reset values at unwanted locations to zero
        top_val = top_val * pop_mask.unsqueeze(-1)
        # a pop must be applied only when it is required and possible
        self._top_cur = self._top_cur - top_succ * pop_mask
        succ = 1 - (1 - top_succ) * pop_mask    # only the errors at the required locations matter
        return top_val, succ

    def dump(self) -> Tuple[torch.Tensor, torch.LongTensor]:
        # top_cur: (batch,)
        lengths = self._top_cur + 1
        max_length = torch.max(lengths)
        if max_length <= 0:
            return (self._storage.new_zeros(self.max_batch_size, 1, self.item_size),
                    self._storage.new_zeros(self.max_batch_size, 1))  # empty stack

        value = self._storage[:, :max_length]
        # row = max_length + 1, including the zero-length case
        mask = self._storage.new_ones(max_length + 1, max_length).tril(-1)[lengths]
        return value, mask


if __name__ == '__main__':
    stack = TensorBatchStack(5, 3, 1)
    stack.push(torch.randn(5, 1), torch.tensor([1, 1, 0, 0, 1]).long())
    stack.push(torch.randn(5, 1), torch.tensor([1, 1, 0, 0, 1]).long())
    stack.push(torch.randn(5, 1), torch.tensor([1, 1, 0, 0, 1]).long())
    stack.push(torch.randn(5, 1), torch.tensor([1, 1, 0, 0, 1]).long())
    stack.pop(torch.tensor([0, 0, 0, 0, 1]).long())
    stack.dump()
    print(list(x.squeeze() for x in stack.dump()))
