from typing import Callable, TypeVar, Any, Optional, Tuple, Mapping, List
from .retriever import Retriever
from functools import partial
from itertools import chain
import pickle
import logging
from collections import defaultdict
from trialbot.data.dataset import Dataset

ExampleType = Any
KeyFuncType = Callable[[ExampleType], Optional[str]]
GroupExpansionFuncType = Callable[[str], List[str]]
SimilarityType = Mapping[str, List[str]]
SimilarityCacheType = Tuple[SimilarityType, SimilarityType, SimilarityType]

class KVCacheRetriever(Retriever):
    def __init__(self,
                 key: KeyFuncType,
                 dataset: Dataset,
                 similarity_cache: SimilarityCacheType,
                 group_expansion: Optional[GroupExpansionFuncType] = None):
        """
        a key value store, key=keyfunc(example), val=dataset.index(example)
        :param key: example -> key_string
        :param dataset: iterable dataset
        :param similarity_cache: tuple of 3 cache, each cache: key_string -> List[key_string]
        """
        super().__init__(dataset)
        self._kvstore = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self.get_example_key = key
        self._sim_cache = similarity_cache
        self.group_expansion = group_expansion

    def build_key_value_store(self):
        """build the KV store with key=intepretable keys, val=dataset list index"""
        if len(self._kvstore) == 0:
            for i, example in enumerate(self.dataset):
                key = self.get_example_key(example)
                if key is None:
                    self.logger.warning("The example do not have a key and is thus IGNORED.")
                    continue

                self._kvstore[self.get_example_key(example)].append(i)

    def search(self, example, example_source: str = "train") -> List[ExampleType]:
        """example -> key_string -> similar key_strings -> dataset indices -> similar examples"""
        similar_keys = self._get_similar_keys(example, example_source)
        similar_ids = list(chain(*map(self._kvstore.get, similar_keys)))
        return self.dataset[similar_ids]

    def search_group(self, example, example_source: str = "train") -> List[ExampleType]:
        """example -> key_string -> similar key_strings -> more key_strings -> dataset indices -> examples"""
        similar_keys = self._get_similar_keys(example, example_source)
        if self.group_expansion is not None:
            expanded_keys = chain.from_iterable(map(self.group_expansion, similar_keys))
            similar_keys = set(expanded_keys)
        similar_ids = list(chain(*map(self._kvstore.get, similar_keys)))
        return self.dataset[similar_ids]

    def _get_similar_keys(self, example, example_source: str) -> List[str]:
        self.build_key_value_store()
        key = self.get_example_key(example)
        if key is None or example_source not in ("train", "dev", "test"):
            self.logger.warning("The example do not have a Key")
            return []

        sim: SimilarityType = self._sim_cache[["train", "dev", "test"].index(example_source)]
        similar_keys = sim[key]
        if similar_keys is not None and len(similar_keys) > 0 and isinstance(similar_keys[0], list):
            similar_keys = similar_keys[0]  # flat the list if possible: [[1, 2, 3]] -> [1, 2, 3]

        return similar_keys


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

def get_hyp_group(key: str, topk=5) -> List[str]:
    ex_id = key.split('-')[0]
    group_keys = [f"{ex_id}-{str(x)}" for x in range(topk)]
    return group_keys

HypIDCacheRetriever = lambda filename, dataset: KVCacheRetriever(
    key=get_hyp_key,
    dataset=dataset,
    similarity_cache=pickle.load(open(filename, 'rb')),
    group_expansion=partial(get_hyp_group, topk=5),
)

