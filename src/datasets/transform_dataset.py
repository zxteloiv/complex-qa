from trialbot.data import CompositionalDataset

class TransformData(CompositionalDataset):
    def __init__(self, dataset, transform_fn):
        super().__init__(dataset)
        self.transform = transform_fn

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i: int):
        example = super().get_example(i)
        if example is None:
            return None
        return self.transform(example)
