from typing import Optional, Tuple
import torch

class BatchedStack:
    def __init__(self, max_batch_size: int, max_stack_size: int = 0):
        self.max_batch_size = max_batch_size
        self.max_stack_size = max_stack_size
        self._lists = None

        self.reset(max_batch_size)

    def reset(self, batch_size):
        self._lists = [[] * batch_size]
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

            if ind_push > 0:
                self._lists[item_id].append(data[item_id])

        return push_mask.new_tensor(succ)

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

    def mixed_action(self, action: torch.Tensor, data: torch.Tensor):
        """
        Execute actions for different items in a batch.
        :param action: (batch,), integer indicators for every item in batch, to push (1), pop (-1) or retain (0).
        :param data: (batch, *), the item to push onto the stack.
        :return:
        """
        raise NotImplementedError
