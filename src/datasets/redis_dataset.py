from .composition_dataset import CompositionalDataset, Dataset
from typing import Optional, Tuple
import redis
import logging
import pickle

class RedisDataset(CompositionalDataset):
    def __init__(self,
                 dataset: Dataset,
                 conn: Optional[Tuple[str, int, int]] = None,
                 prefix: str = "",
                 expire_sec: int = 0,
                 ):
        super().__init__(dataset)
        host, port, db = conn
        pool = redis.ConnectionPool(host=host, port=port, db=db)
        self.r = redis.Redis(connection_pool=pool)
        self.prefix = prefix if prefix.endswith('_') else prefix + '_'
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset = dataset
        self.expire_sec = expire_sec

    def _set(self, k, v):
        arr = pickle.dumps(v)
        self.r.set(k, arr, ex=self.expire_sec if self.expire_sec > 0 else None)

    def _read(self, k):
        v = self.r.get(k)
        if v is not None:
            v = pickle.loads(v)
        return v

    def get_example(self, i: int):
        obj = self._read(self.prefix + str(i))
        if obj is None:
            obj = self.dataset.get_example(i)

        self._set(self.prefix + str(i), obj)
        return obj

    def __len__(self):
        l = self._read(self.prefix + 'len')
        if l is None:
            l = len(self.dataset)
        self._set(self.prefix + 'len', l)
        return l

