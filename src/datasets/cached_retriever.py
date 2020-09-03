from typing import Callable, TypeVar, Any, Optional, Tuple, Mapping, List
from .retriever import Retriever
from functools import partial
from itertools import chain
import pickle
import logging
from collections import defaultdict, Counter
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
                 grouper: Optional['Grouper'] = None):
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
        self.grouper = grouper

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
        similar_keys = self.grouper(similar_keys)
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


def get_hyp_key(example: ExampleType) -> Optional[str]:
    ex_id, hyp_rank = list(map(example.get, ('ex_id', 'hyp_rank')))
    if ex_id is None or hyp_rank is None:
        return None
    return f"{ex_id}-{hyp_rank}"

class Grouper:
    def __init__(self,
                 ex2group_func: Callable[[str], str] = None,
                 group_expansion_func: Callable[[str], List[str]] = None,
                 ):
        self._ex2group = ex2group_func
        self._group_exp = group_expansion_func

    def get_group_id_from_example_key(self, example_key: str) -> str:
        if self._ex2group is not None:
            return self._ex2group(example_key)

        raise NotImplementedError

    def expand_group(self, group_key: str) -> List[str]:
        if self._group_exp is not None:
            return self._group_exp(group_key)

        raise NotImplementedError

    def __call__(self, keys: List[str]) -> List[str]:
        group_ids = map(self.get_group_id_from_example_key, filter(None, keys))
        # voting and sorting
        counter = Counter(group_ids)
        groups_desc: List[str] = sorted(counter.keys(), key=lambda k: counter[k], reverse=True)
        expanded_keys = list(chain.from_iterable(self.expand_group(g) for g in groups_desc))

        return expanded_keys

hyp_grouper = Grouper(ex2group_func=(lambda exkey: exkey.split('-')[0]),
                      group_expansion_func=lambda gkey: [f"{gkey}-{i}" for i in range(5)])

id_grouper = Grouper(ex2group_func=lambda k: k, group_expansion_func=lambda k: [k])

IDCacheRetriever = lambda filename, dataset: KVCacheRetriever(
    key=lambda example: example.get('ex_id'),
    dataset=dataset,
    similarity_cache=pickle.load(open(filename, 'rb')),
    grouper=id_grouper,
)

HypIDCacheRetriever = lambda filename, dataset: KVCacheRetriever(
    key=get_hyp_key,
    dataset=dataset,
    similarity_cache=pickle.load(open(filename, 'rb')),
    grouper=hyp_grouper,
)

