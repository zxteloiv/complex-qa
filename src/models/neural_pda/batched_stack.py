from typing import Optional, Tuple
import torch

class BatchStack:
    def reset(self, batch_size, default_device = None):
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

    def pop(self,
            batch_size: Optional[int] = None,
            default_item: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remove the top item from each stack, and return them.
        :param batch_size: the required batch to pop
        :param default_item: the item to fill if the stack is empty.
        :return: A tuple of the two
                (batch, *) tuple of data items popped out,
                (batch,) an errcode for each item, (1) for successful pop, (0) if the appropriate stack is empty.
        """
        return self.top(pop_data=True, batch_size=batch_size, default_item=default_item)

    def top(self,
            pop_data: bool = False,
            batch_size: Optional[int] = None,
            default_item: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top item from each stack.
        :param pop_data: not only get the top item of each stack, but also remove them from the stack top.
        :param batch_size: the required batch to pop
        :param default_item: the item to fill if the stack is empty.
        :return: A tuple of the two
                (batch, *) tuple of data items popped out,
                (batch,) an errcode for each item, (1) for successful pop, (0) if the appropriate stack is empty.
        """
        raise NotImplementedError


class ListedBatchStack(BatchStack):
    def __init__(self, max_batch_size: int, max_stack_size: int = 0):
        self.max_batch_size = max_batch_size
        self.max_stack_size = max_stack_size
        self._lists = None

        self.reset(max_batch_size, None)

    def reset(self, batch_size, default_device=None):
        self._lists = [[] for _ in range(batch_size)]
        self.max_batch_size = batch_size

    def push(self, data: torch.Tensor, push_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        :param data: (batch, *) item to push onto the stack
        :param push_mask: (batch,) indicators to push items
                (1) onto the stack or not to (0) for each item in the batch
        :return: the success indicators for the push action,
                (1) if successful (either pushed or retained), (0) if max_stack_size is exceeded.
        """
        succ = []
        for item_id, ind_push in enumerate(push_mask):
            # possible errors when pushing
            if len(self._lists) <= item_id: # size of the current batch exceeds the max batch size.
                succ.append(0)
            elif 0 < self.max_stack_size == len(self._lists[item_id]): # stack has a valid and full size limit
                succ.append(0)
            else:
                succ.append(1)

            # even though the push indicator is zero, the success flag is also set 1
            if ind_push > 0:
                self._lists[item_id].append(data[item_id])

        return push_mask.new_tensor(succ)

    def top(self,
            pop_data: bool = False,
            batch_size: Optional[int] = None,
            default_item: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top item from each stack.
        :param pop_data: not only get the top item of each stack, but also remove them from the stack top.
        :param batch_size: the required batch to pop
        :param default_item: the item to fill if the stack is empty.
        :return: A tuple of the two
                (batch, *) tuple of data items popped out,
                (batch,) an errcode for each item, (1) for successful pop, (0) if the appropriate stack is empty.
        """
        batch_size = batch_size or self.max_batch_size
        assert batch_size <= self.max_batch_size and batch_size <= len(self._lists)

        # the first item available in all lists is chosen as the default, in case that the default item is not given
        if default_item is None:
            default_item = next(x for l in self._lists for x in l)

        data = []
        succ = []
        for item_id in range(batch_size):
            item_stack = self._lists[item_id]
            if len(item_stack) == 0:
                succ.append(0)
                data.append(torch.zeros_like(default_item))
            else:
                succ.append(1)
                data.append(item_stack.pop() if pop_data else item_stack[-1])

        data = torch.stack(data)
        succ = data.new_tensor(succ)
        return data, succ


class TensorBatchStack(BatchStack):
    def __init__(self,
                 max_batch_size: int,
                 max_stack_size: int = 0,
                 item_size: int = 1,
                 ):
        self.max_batch_size = max_batch_size
        self.max_stack_size = max_stack_size
        self.item_size = item_size
        self._storage: Optional[torch.Tensor] = None
        self._top_cur: Optional[torch.Tensor] = None
        self.reset(max_batch_size)
        self.inplace = False

    def reset(self, batch_size, default_device: Optional[torch.device] = None):
        self._storage = torch.zeros(batch_size, self.max_stack_size, self.item_size, device=default_device)
        self._top_cur = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=default_device)
        self.max_batch_size = batch_size

    def push(self, data: torch.Tensor, push_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        :param data: (batch, *) item to push onto the stack
        :param push_mask: (batch,)
                indicators to push items (1) onto the stack or not to (0) for each item in the batch
        :return: the success indicators for the push action,
                (1) if successful (either pushed or retained), (0) if max_stack_size is exceeded.
        """
        batch_sz = data.size()[0]
        assert push_mask.size()[0] == batch_sz
        # force the incoming batch size equal to the storage,
        # such that no slicing is required and the implementation is easier.
        assert batch_sz == self.max_batch_size

        # stack error is only triggered when the push action is required but the stack is full.
        # stack_error: (batch,)
        stack_error = (self._top_cur >= (self.max_stack_size - 1)) * push_mask
        succ = 1 - stack_error

        batch_range = torch.arange(batch_sz, device=data.device)
        # increasing the stack top cursor only if the push action is valid
        self._top_cur = self._top_cur + 1 * succ.long()
        val_backup = self._storage[batch_range, self._top_cur]
        push_mask = push_mask.unsqueeze(-1)
        val_new = val_backup * (1 - push_mask) + data * push_mask

        new_storage = self._storage.clone()
        new_storage[batch_range, self._top_cur] = val_new
        self._storage = new_storage
        return succ

    def top(self,
            pop_data: bool = False,
            batch_size: Optional[int] = None,
            default_item: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top item from each stack.
        :param pop_data: not only get the top item of each stack, but also remove them from the stack top.
        :param batch_size: the required batch to pop, must be the max_batch_size for a tensor-based BatchStack.
        :param default_item: the item to fill if the stack is empty.
        :return: A tuple of the two
                (batch, *) tuple of data items popped out,
                (batch,) an errcode for each item, (1) for successful pop, (0) if the appropriate stack is empty.
        """
        assert batch_size is None or batch_size == self.max_batch_size
        batch_size = self.max_batch_size

        device = self._storage.device
        if default_item is None:
            default_item = torch.zeros((self.max_batch_size, self.item_size), device=device)

        # succ, top_idx: (batch,)
        succ = ((self._top_cur >= 0) * (self._top_cur < self.max_stack_size)).long()
        top_idx = self._top_cur * succ  # invalid position has the default index 0

        # batch_range: (batch,)
        # val_mask: (batch, 1)
        batch_range = torch.arange(self.max_batch_size, device=device)
        val_mask = succ.unsqueeze(-1)
        top_val = self._storage[batch_range, top_idx] * val_mask + (1 - val_mask) * default_item

        if pop_data:
            self._top_cur = self._top_cur - 1 * succ

        return top_val, succ

if __name__ == '__main__':
    stack = TensorBatchStack(10, 3, 5)
    _, succ = stack.top(batch_size=10)
    assert (succ == torch.zeros(10)).all()
    print("empty stack:", stack._top_cur)

    # first push
    data = torch.randint(100, 200, (10, 5))
    succ = stack.push(data, torch.ones(10, dtype=torch.long))
    assert (succ == torch.ones(10)).all()
    print("after 1st push:", stack._top_cur)

    # read the value from the first push
    val, succ = stack.top(batch_size=10)
    assert (val == data).all()
    assert (succ == torch.ones(10)).all()
    print("read from the 1st push:", stack._top_cur)

    # second push
    succ = stack.push(data, torch.ones(10, dtype=torch.long))
    assert (succ == torch.ones(10)).all()
    print("2nd push:", stack._top_cur)
    # third push
    succ = stack.push(data, torch.ones(10, dtype=torch.long))
    assert (succ == torch.ones(10)).all()
    print("3rd push:", stack._top_cur)
    # forth push must be failed for the entire batch
    succ = stack.push(data, torch.ones(10, dtype=torch.long))
    assert (succ == torch.zeros(10)).all()
    print("4rd push:", stack._top_cur)

    # pop out then, must be successful
    val, succ = stack.pop(batch_size=10)
    assert (val == data).all()
    assert (succ == torch.ones(10)).all()
    print("pop out:", stack._top_cur)

    # forth push again, which would be successful
    succ = stack.push(data, torch.ones(10, dtype=torch.long))
    assert (succ == torch.ones(10)).all()
    print("4rd push again:", stack._top_cur)

    # pop out for 3 times
    val, succ = stack.pop(batch_size=10)
    assert (val == data).all()
    assert (succ == torch.ones(10)).all()
    print("1st pop:", stack._top_cur)
    val, succ = stack.pop(batch_size=10)
    assert (val == data).all()
    assert (succ == torch.ones(10)).all()
    print("2nd pop:", stack._top_cur)
    val, succ = stack.pop(batch_size=10)
    assert (val == data).all()
    assert (succ == torch.ones(10)).all()
    print("3rd pop:", stack._top_cur)

    # read value from empty stack -> must be failed
    _, succ = stack.top(batch_size=10)
    assert (succ == torch.zeros(10)).all()
    print("empty stack again:", stack._top_cur)

