import time

import requests
import sys

BASEURL = 'https://api.semanticscholar.org/graph/v1'


def load_s2_api_key() -> str:
    import os
    key = os.getenv('S2_API_KEY')
    if key is None:
        raise EnvironmentError('API key not found.')
    return key


def search_ss_title(title: str) -> list:
    def ss_norm(title: str) -> str:
        s = title.replace('{', '').replace('}', '').replace('\\', '')
        return s
        # return urllib.parse.quote(s)

    url = f'{BASEURL}/paper/search'
    params = {
        'limit': 3,
        'fields': 'title,year,paperId,corpusId,authors,venue,abstract,citationCount,tldr,externalIds',
        'query': ss_norm(title),
    }
    headers = {
        'x-api-key': load_s2_api_key(),
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        jsonobj = response.json()['data']
        return jsonobj
    except:
        print(f'failed to search title: {title}', file=sys.stderr)
        return []


_last_doi_time = time.time()


def search_ss_doi(doi: str) -> dict:
    global _last_doi_time

    url = f'{BASEURL}/paper/DOI:{doi}'
    params = {
        'fields': 'title,year,paperId,corpusId,authors,venue,abstract,citationCount,tldr,externalIds',
    }
    headers = {
        'x-api-key': load_s2_api_key(),
    }

    req_time = time.time()
    if req_time - _last_doi_time <= 0.15:
        time.sleep(0.15 - req_time + _last_doi_time)
        _last_doi_time = req_time

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        jsonobj = response.json()
        return jsonobj

    except:
        print(f'failed to search doi: {doi}', file=sys.stderr)
        return {}



