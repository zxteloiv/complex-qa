from trialbot.data.dataset import Dataset
class Retriever:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def search(self, example) -> list:
        """
        Search similar result for the example

        :param example: an instance
        :return: a list of similar instances from the training set for the given example
        """
        raise NotImplementedError
