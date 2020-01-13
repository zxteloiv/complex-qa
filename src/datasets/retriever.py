from typing import List
from trialbot.data.dataset import Dataset

class Retriever:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def search(self, example, example_source: str) -> list:
        """
        Search similar result for the example

        :param example: an instance
        :param example_source: the source of the instance, must be "train", "dev" or "test".
        :return: a list of similar instances from the training set for the given example
        """
        raise NotImplementedError

class MultiRetriever:
    def __init__(self, retrievers: List[Retriever]):
        self.retrievers = retrievers

    def search(self, example, example_source):
        res = []
        for ret in self.retrievers:
            res.extend(ret.search(example, example_source))
        return res
