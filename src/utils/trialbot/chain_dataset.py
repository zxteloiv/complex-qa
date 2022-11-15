from typing import List

from trialbot.data.dataset import Dataset


class ChainDataset(Dataset):
    def __init__(self, chained_datasets: List[Dataset], lengths=None):
        super().__init__()
        self.chained_datasets = chained_datasets
        self.lengths = lengths

    def load_lengths(self):
        if self.lengths is None:
            self.lengths = [len(ds) for ds in self.chained_datasets]

    def __len__(self):
        self.load_lengths()
        return sum(self.lengths)

    def get_example(self, i: int):
        self.load_lengths()
        for ds, length in zip(self.chained_datasets, self.lengths):
            if i < length:
                return ds[i]
            else:
                i = i - length

        raise IndexError
