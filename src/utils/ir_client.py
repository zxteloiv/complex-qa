from typing import Iterable, Dict, Union, List, Any, Tuple
import logging
import re


class VolatileBM25Index:
    def __init__(self, default_search_limit: int = 10):
        import sqlite3
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.debug('building index...')
        self.conn = sqlite3.connect(':memory:')
        self.conn.execute('create virtual table kvmem using fts5(key, payload);')
        self.conn.commit()
        self.limit: int = default_search_limit

    def indexing(self, key: Union[str, List[str]], payload: Union[None, Any, List[Any]]):
        if payload is None:
            self.logger.warning('No payload given, default to the integer 0')
            payload = 0 if isinstance(key, str) else [0 for _ in range(len(key))]

        if isinstance(key, str):
            self.logger.debug(f'inserting item {key} into index...')
            self.conn.execute('insert into kvmem values (?, ?);', (key, payload))

        elif isinstance(key, list):
            data = list(zip(key, payload))
            self.logger.debug('inserting %d items into index...' % len(key))
            self.conn.executemany('insert into kvmem values (?, ?);', data)

        else:
            raise TypeError('indexing key must be str of list of str')

        self.conn.commit()

    def search_index(self, key: str, limit: int = None) -> List[Tuple[str, Any]]:
        limit = limit or self.limit

        keywords = []
        for k in set(key.split()):
            k = k.replace('"', '')
            keywords.append(f'"{k}"')

        fts_str = ' OR '.join(keywords)
        cur = self.conn.execute(
            f'select `key`, payload from kvmem where `key` match (?)'
            f'order by bm25(kvmem) limit {limit}',
            (fts_str,)
        )
        return cur.fetchall()   # list of tuples of (key str, payload)

    @staticmethod
    def from_data_list(keys: List[str], payloads: List[Any], default_search_limit: int = 10):
        idx = VolatileBM25Index(default_search_limit)
        idx.indexing(keys, payloads)
        return idx


class SolrClient:
    def __init__(self, core, host: str = "localhost", port: int = 8975):
        self.core = core
        self.host = host
        self.port = port
        # import requests
        # self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

    def indexing(self, *args, **kwargs):
        self.logger.info("no need to index documents")

    def search(self, query: dict):
        if query.get('wt') is None:
            query['wt'] = 'json'
        if query.get('df') is None:
            query['df'] = 'hyp'

        import requests

        # response = self.session.post(f'http://{self.host}:{self.port}/solr/{self.core}/query', data=query)
        response = requests.post(f'http://{self.host}:{self.port}/solr/{self.core}/query', data=query)
        return response.json()

    @staticmethod
    def escape(query_text: str):
        pat = re.compile(r'[\+\-&\|\!\(\)\{\}\[\]\^"\~\*\?\:\/]')
        return pat.sub(r'\\\g<0>', query_text)

