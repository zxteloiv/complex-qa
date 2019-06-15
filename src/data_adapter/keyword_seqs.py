from typing import Iterable, Dict, List

import os.path
import logging
import allennlp.data.dataset_readers
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer

class CharKeywordReader(allennlp.data.DatasetReader):
    def __init__(self, max_sent_length=30, topic_model_dir="~/.complex_qa/topic_model"):
        super(CharKeywordReader, self).__init__()
        self.max_sent_length = max_sent_length
        import jieba
        self._cut = lambda x: [w.strip() for w in jieba.cut(x) if w not in ("", ' ', '\t')]
        from utils.keywords_extraction.lda import topic_model as TopicModel
        topic_model_path = os.path.join(os.path.expanduser(topic_model_dir), 'topic.model')
        logging.info('Loading topic model from: %s' % topic_model_path)
        self._topic_model = TopicModel(topic_model_path)

    def _read(self, file_path: str) -> Iterable[Instance]:
        for line in open(file_path):
            parts = line.rstrip().split('\t')
            if len(parts) != 2:
                logging.warning('Skip invalid line: %s' % str(parts))
                continue
            src, tgt = parts
            keywords = self.get_keywords(src) + self.get_keywords(tgt)
            yield self.text_to_instance(src, tgt, keywords)

    def text_to_instance(self, src: str, tgt: str, keywords: List[str]):
        def filter_split_str(sent: str) -> List[str]:
            return list(filter(lambda x: x not in ("", ' ', '\t'), list(sent)))

        src = filter_split_str(src)[:self.max_sent_length]
        src_fld = TextField(list(map(Token, src)), {'tokens': SingleIdTokenIndexer()})

        tgt = filter_split_str(tgt)[:self.max_sent_length]
        tgt_fld = TextField(list(map(Token, tgt)), {'tokens': SingleIdTokenIndexer()})

        keywords = filter_split_str("".join(keywords))
        keyword_fld = TextField(list(map(Token, keywords)), {'tokens': SingleIdTokenIndexer()})

        instance = Instance({
            "source_tokens": src_fld,
            "target_tokens": tgt_fld,
            "keyword_tokens": keyword_fld,
        })
        return instance

    def get_keywords(self, text: str, num=3) -> List[str]:
        text = " ".join(self._cut(text))
        keywords_dict = self._topic_model.get_doc_keywords(text, num)
        keys = sorted(keywords_dict.keys(), key=lambda x: keywords_dict[x])
        return keys[:num]

class CharExtractedKeywordReader(CharKeywordReader):
    def __init__(self, max_sent_length=30, topic_model_dir="~/.complex_qa/topic_model"):
        super(CharExtractedKeywordReader, self).__init__()

    def _read(self, file_path: str) -> Iterable[Instance]:
        for line in open(file_path):
            parts = line.rstrip().split('\t')
            if len(parts) != 3:
                logging.warning('Skip invalid line: %s' % str(parts))
                continue

            src, tgt, keywords = parts
            keywords = keywords.split(',')
            yield self.text_to_instance(src, tgt, keywords)
