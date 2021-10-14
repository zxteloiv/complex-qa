from typing import Any
from trialbot.data import RedisDataset

class WritableRedis(RedisDataset):
    def __setitem__(self, index: int, value: Any) -> None:
        self._set(self.prefix + str(index), value)