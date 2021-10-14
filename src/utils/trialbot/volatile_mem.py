from typing import Any
from trialbot.data.dataset import CompositionalDataset


class VolatileDataset(CompositionalDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        # key protocol: i -> example
        self.writable = {}

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i: int):
        if i in self.writable:
            return self.writable[i]

        example = self.dataset.get_example(i)
        self.writable[i] = example
        return example

    def __setitem__(self, index: int, value: Any) -> None:
        self.writable[index] = value



