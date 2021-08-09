from trialbot.data import CompositionalDataset

class IDSupplement(CompositionalDataset):
    def __init__(self, dataset, key='id'):
        super().__init__(dataset)
        self.key = key

    def get_example(self, i: int):
        data = self.dataset.get_example(i)
        data[self.key] = i
        return data

    def __len__(self):
        return len(self.dataset)