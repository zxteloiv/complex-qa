from typing import Iterable, Dict, List, Tuple

import allennlp.data

import re
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer

class MSParSDatasetReader(allennlp.data.DatasetReader):
    def __init__(self):
        super(MSParSDatasetReader, self).__init__()

    def _read(self, file_path: str) -> Iterable[Instance]:
        raw_example = []
        for i, l in enumerate(open(file_path)):
            ord_i = i + 1
            if ord_i % 5 != 0:
                raw_example.append(l)
                continue

            yield self.text_to_instance(raw_example)

    def text_to_instance(self, example: List) -> Instance:
        qid = re.search(r' id=(\d+)>\t', example[0]).group(1)
        raw_q, raw_lf, raw_params, raw_qtype = map(lambda x: x.split('\t')[1], example[:4])

        qs = list(map(lambda x: x.split(' '), raw_q.split('|||')))
        lfs = list(map(lambda x: x.split(' '), raw_lf.split('|||')))

        ins = Instance({
            "question": ListField([
                TextField(list(map(Token, q)), {"tokens": SingleIdTokenIndexer('src_tokens')})
                for q in qs
            ]),
            "lf": ListField([
                TextField(list(map(Token, lf)), {"tokens": SingleIdTokenIndexer('tgt_tokens')})
                for lf in lfs
            ])
        })

        return ins


