from typing import Iterable, Dict, List, Tuple

import allennlp.data.dataset_readers

from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL


class TabSepDatasetReader(allennlp.data.DatasetReader):
    def __init__(self, lazy=False):
        super(TabSepDatasetReader, self).__init__(lazy)
        self.instance_keys = ("source_tokens", "target_tokens")

    def _read(self, file_path: str) -> Iterable[Instance]:
        for line in open(file_path):
            src, tgt = line.rstrip().split('\t')
            instance = self.text_to_instance(src.split(' '), tgt.split(' '))
            yield instance

    def text_to_instance(self, source: List[str], target: List[str]) -> Instance:

        x = TextField(list(map(Token, source)),
                      {'tokens': SingleIdTokenIndexer('src_tokens')})
        y = TextField(list(map(Token, [START_SYMBOL] + target + [END_SYMBOL])),
                      {'tokens': SingleIdTokenIndexer('tgt_tokens')})
        instance = Instance(dict(zip(self.instance_keys, (x, y))))

        return instance

class TabSepJiebaCutReader(TabSepDatasetReader):
    def __init__(self, lazy=False):
        super(TabSepJiebaCutReader, self).__init__(lazy)
        import jieba
        self._cut = lambda x: [w.strip() for w in jieba.cut(x)]

    def _read(self, file_path: str):
        for line in open(file_path):
            src, tgt = line.rstrip().split('\t')
            instance = self.text_to_instance(self._cut(src), self._cut(tgt))
            yield instance





