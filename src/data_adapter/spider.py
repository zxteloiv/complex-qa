from typing import Iterable, Dict, List, Tuple
import json

import allennlp.data.dataset_readers

from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL


class SpiderDatasetReader(allennlp.data.DatasetReader):
    def __init__(self, lazy=False, no_value=True):
        super(SpiderDatasetReader, self).__init__(lazy)
        self.no_value = no_value

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset = json.load(open(file_path))
        for example in dataset:
            question = example['question_toks']
            query_toks_no_value = example['query_toks_no_value']
            query_toks = example['query_toks']
            query = query_toks_no_value if self.no_value else query_toks
            instance = self.text_to_instance(question, query)

            yield instance

    def text_to_instance(self, question: List[str], logic_form: List[str]) -> Instance:

        x = TextField(list(map(Token, question)),
                      {'tokens': SingleIdTokenIndexer('src_tokens')})
        z = TextField(list(map(Token, [START_SYMBOL] + logic_form + [END_SYMBOL])),
                      {'tokens': SingleIdTokenIndexer('tgt_tokens')})
        instance = Instance(dict(source_tokens=x, target_tokens=z))

        return instance


