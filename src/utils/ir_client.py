from typing import Iterable, Dict
import logging
import re

class RetrievalIndex:
    def __init__(self, schema=None, idx_name=None, use_ram_storage=True):
        from whoosh.fields import Schema, NGRAM, STORED
        from whoosh.filedb.filestore import RamStorage, FileStorage
        from whoosh.qparser import QueryParser, OrGroup

        idx_name = idx_name or __name__
        storage = RamStorage() if use_ram_storage else FileStorage(idx_name)
        self.is_default_schema = schema is None
        schema = schema or Schema(key=NGRAM(maxsize=5), value=STORED())
        idx = storage.create_index(schema, idx_name)
        self.idx = idx
        self.parser = QueryParser("key", schema, group=OrGroup)

    def indexing(self, dataset: Iterable[Dict[str, str]]):
        writer = self.idx.writer()
        for e in dataset:
            writer.add_document(**e)
        writer.commit()

    def search(self, key, **kwargs):
        with self.idx.searcher() as searcher:
            query = self.parser.parse(key)
            res = searcher.search(query, **kwargs)
            if self.is_default_schema:
                return [r['value'] for r in res]
            else:
                return res


class SolrClient:
    def __init__(self, core, host: str = "localhost", port: int = 8975):
        self.core = core
        self.host = host
        self.port = port
        import requests
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

    def indexing(self, *args, **kwargs):
        self.logger.info("no need to index documents")

    def search(self, query: dict):
        if query.get('wt') is None:
            query['wt'] = 'json'
        if query.get('df') is None:
            query['df'] = 'hyp'

        response = self.session.post(f'http://{self.host}:{self.port}/solr/{self.core}/query', data=query)
        return response.json()

    @staticmethod
    def escape(query_text: str):
        pat = re.compile(r'[\+\-&\|\!\(\)\{\}\[\]\^"\~\*\?\:\/]')
        return pat.sub(r'\\\g<0>', query_text)

