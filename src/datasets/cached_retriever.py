
from .retriever import Retriever
import pickle
import logging
from collections import defaultdict

class IDCacheRetriever(Retriever):
    """
    IDCacheRetriever.
    Training set are indexed by ids.
    Similar examples for a given one could be directly retrieved via id.
    """
    def __init__(self, filename, dataset):
        super().__init__(dataset)
        matrices = pickle.load(open(filename, 'rb'))
        self._sim_matrics = matrices
        self.logger = logging.getLogger(__name__)

        # natural language input are indexed by the exid only
        self._eid_to_examples = defaultdict(list)

    def build_example_index(self):
        if len(self._eid_to_examples) == 0:
            for i, example in enumerate(self.dataset):
                self._eid_to_examples[example["ex_id"]].append(i)

    def search(self, example, example_source: str = "train"):
        self.build_example_index()

        eid = example.get("ex_id")
        if eid is None or example_source not in ("train", "dev", "test"):
            self.logger.warning("Given example do not have an ID.")
            return []
        eid = int(eid)

        m = self._sim_matrics[["train", "dev", "test"].index(example_source)]
        similar_example_ids = []
        for similar_eid in m[eid]:
            similar_example_ids += self._eid_to_examples[similar_eid]

        return self.dataset[similar_example_ids]

class HypIDCacheRetriever(Retriever):
    """
    IDCacheRetriever.
    Training set are indexed by ids.
    Similar examples for a given one could be directly retrieved via id.
    """
    def __init__(self, filenames, dataset):
        super().__init__(dataset)
        matrices = [pickle.load(open(f, 'rb')) for f in filenames]
        self._sim_matrics = matrices
        self.logger = logging.getLogger(__name__)

        # natural language input are indexed by the exid only
        self._eid_to_examples = defaultdict(list)

    def build_example_index(self):
        if len(self._eid_to_examples) == 0:
            for i, example in enumerate(self.dataset):
                self._eid_to_examples[example["ex_id"]].append(i)

    def search(self, example, example_source: str = "train"):
        self.build_example_index()

        eid, hyp_rank = list(map(example.get, ("ex_id", "hyp_rank")))
        if eid is None or hyp_rank is None or example_source not in ("train", "dev", "test"):
            self.logger.warning("Given example do not have an ID.")
            return []
        eid = int(eid)
        hyp_rank = int(hyp_rank)
        query_key = f"{eid}-{hyp_rank}"

        m = self._sim_matrics[["train", "dev", "test"].index(example_source)]
        if query_key not in m:
            return []

        similar_example_ids = []
        for similar_eid in m[query_key]:
            similar_example_ids += self._eid_to_examples[similar_eid]

        return self.dataset[similar_example_ids]

class SimRetriever(Retriever):
    def __init__(self, nlfile, lffiles, dataset):
        super().__init__(dataset)
        self.logger = logging.getLogger(__name__)

        self._sim_nl = pickle.load(open(nlfile, 'rb'))
        self._sim_lf = [pickle.load(open(f, 'rb')) for f in lffiles]

        # natural language input are indexed by the exid only
        self._eid_to_examples = defaultdict(list)

    def build_example_index(self):
        if len(self._eid_to_examples) == 0:
            for i, example in enumerate(self.dataset):
                self._eid_to_examples[example["ex_id"]].append(i)

    def _search_lf(self, example, example_source: str = "train"):
        self.build_example_index()

        eid, hyp_rank = list(map(example.get, ("ex_id", "hyp_rank")))
        if eid is None or hyp_rank is None or example_source not in ("train", "dev", "test"):
            self.logger.warning("Given example do not have an ID.")
            return []
        eid = int(eid)
        hyp_rank = int(hyp_rank)
        query_key = f"{eid}-{hyp_rank}"

        m = self._sim_lf[["train", "dev", "test"].index(example_source)]
        if query_key not in m:
            return []

        similar_example_ids = []
        for similar_eid in m[query_key]:
            similar_example_ids += self._eid_to_examples[similar_eid]

        return self.dataset[similar_example_ids]

    def _search_nl(self, example, example_source: str = "train"):
        self.build_example_index()

        eid = example.get("ex_id")
        if eid is None or example_source not in ("train", "dev", "test"):
            self.logger.warning("Given example do not have an ID.")
            return []
        eid = int(eid)

        m = self._sim_nl[["train", "dev", "test"].index(example_source)]
        similar_example_ids = []
        for similar_eid in m[eid]:
            similar_example_ids += self._eid_to_examples[similar_eid]

        return self.dataset[similar_example_ids]

    def search(self, example, example_source: str = "train"):
        return self._search_lf(example, example_source) + self._search_nl(example, example_source)
