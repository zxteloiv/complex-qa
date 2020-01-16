from typing import Callable, TypeVar, Any, Optional, Tuple, Mapping, List
from .retriever import Retriever
from functools import reduce
import pickle
import logging
from collections import defaultdict
from trialbot.data.dataset import Dataset

ExampleType = Any
KeyFuncType = Callable[[ExampleType], Optional[str]]
SimilarityType = Mapping[str, List[str]]
SimilarityCacheType = Tuple[SimilarityType, SimilarityType, SimilarityType]

class KVCacheRetriever(Retriever):
    def __init__(self, key: KeyFuncType, dataset: Dataset, similarity_cache: SimilarityCacheType):
        super().__init__(dataset)
        self._kvstore = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self.get_example_key = key
        self._sim_cache = similarity_cache

    def build_key_value_store(self):
        if len(self._kvstore) == 0:
            for i, example in enumerate(self.dataset):
                key = self.get_example_key(example)
                if key is None:
                    self.logger.warning("The example do not have a key and is thus IGNORED.")
                    continue

                self._kvstore[self.get_example_key(example)].append(i)

    def search(self, example, example_source: str = "train"):
        self.build_key_value_store()
        key = self.get_example_key(example)
        if key is None or example_source not in ("train", "dev", "test"):
            self.logger.warning("The example do not have a Key")
            return []

        sim: SimilarityType = self._sim_cache[["train", "dev", "test"].index(example_source)]
        similar_keys = sim[key]
        if len(similar_keys) > 0 and isinstance(similar_keys[0], list):
            similar_keys = similar_keys[0]  # flat the list if possible: [[1, 2, 3]] -> [1, 2, 3]
        similar_ids = reduce(lambda x, y: x + y, (self._kvstore[k] for k in similar_keys), [])
        return self.dataset[similar_ids]

IDCacheRetriever = lambda filename, dataset: KVCacheRetriever(
    key=lambda example: example.get('ex_id'),
    dataset=dataset,
    similarity_cache=pickle.load(open(filename, 'rb')),
)

def get_hyp_key(example: ExampleType) -> Optional[str]:
    ex_id, hyp_rank = list(map(example.get, ('ex_id', 'hyp_rank')))
    if ex_id is None or hyp_rank is None:
        return None
    return f"{ex_id}-{hyp_rank}"

HypIDCacheRetriever = lambda filename, dataset: KVCacheRetriever(
    key=get_hyp_key,
    dataset=dataset,
    similarity_cache=pickle.load(open(filename, 'rb')),
)

