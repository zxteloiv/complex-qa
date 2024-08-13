import dataclasses
import gzip
import json
import sys
import time
import urllib.parse
from dataclasses import dataclass

import requests
import pickle
import trialbot.utils.prepend_pythonpath  # noqa
import sqlite3
import os.path as osp
from pybtex.database import parse_file, BibliographyData, Entry as BibEntry, parse_string

from search_ss import search_ss_title, search_ss_doi
from echo import filter_rule, print_s2entry

bibtex_source = [
    # '~/research/bib/sp-from-anthology.bib',
    # '~/research/bib/zotero-semantic.bib',
    # '~/research/bib/zotero-semantic-parsing.bib',
    # '~/research/bib/Conferences.bib',
    osp.expanduser('~/research/bib/anthology+abstracts.bib.gz'),
]

title_source = [
    osp.expanduser('~/research/bib/gs-sp-titles.txt'),
]


def read_bib(filename: str) -> BibliographyData:
    if filename.endswith(".gz"):
        fread = gzip.open(filename, 'r')
        data = fread.read().decode('utf-8')
    elif filename.endswith('.bib'):
        data = open(filename).read()
    else:
        raise ValueError("Unsupported file type: %s" % filename)

    bib: BibliographyData = parse_string(data, 'bibtex')
    return bib


def bibmain():
    for filename in bibtex_source:
        bib = read_bib(filename)
        for e in bib.entries.values():
            e: BibEntry
            if not filter_bib(e):
                continue

            doi = e.fields.get('DOI', '') or e.fields.get('doi', '')
            if doi:
                pub = search_ss_doi(doi)
                if 'paperId' in pub and filter_rule(pub.get('title', ''), pub):
                    print_s2entry(pub, query='')
                    continue

            print_bib_like_s2(e)


def print_bib_like_s2(e: BibEntry):
    title = e.fields['title']
    try:
        year = int(e.fields['year'])
    except:
        year = 0

    venue = e.fields.get('booktitle') or e.fields.get('journal', None)
    abstract = e.fields.get('abstract')

    try:
        authors = ','.join(str(p) for p in e.persons['author'])
    except:
        authors = ''

    doi = e.fields.get('DOI') or e.fields.get('doi', '')

    print('\t'.join(str(x).replace('\t', ' ').replace('\n', ' ').replace('\r', '') for x in [
        '', '', doi, '',
        title, title, year, venue, authors,
        '', abstract
    ]))


def filter_bib(e: BibEntry) -> bool:
    title = e.fields['title']

    try:
        year = int(e.fields['year'])
    except:
        year = 0

    abstract = e.fields.get('abstract', '')

    if 0 < year < 2020:
        return False

    txt_fields = (title, abstract)

    forbidden_words = ('scene pars', 'semantic segm', 'image segm')
    if any(w in f.lower() for w in forbidden_words for f in txt_fields if f is not None):
        return False

    lower_words = ('semantic pars', 'meaning repr', 'text-to-sql', 'code gen', 'python', 'framenet',
                   'frame semantic', 'knowledge-', 'knowledge ba', 'semantic role')
    words = ('AMR', 'UCCA', 'BMR', 'KBQA', 'SRL')

    if (
            any(w in f.lower() for w in lower_words for f in txt_fields if f is not None)
            or
            any(w in f for w in words for f in txt_fields if f is not None)
    ):
        return True

    return False


def csvmain():
    lasttime = time.time()

    for file in title_source:
        print('reading {}'.format(file), file=sys.stderr)
        for i, t in enumerate(open(file)):
            pubs = search_ss_title(t.strip())

            res = {'title': t.strip(), 'elist': pubs}
            print(json.dumps(res))

            if i % 10 == 0:
                print("{} entries processed".format(i + 1), file=sys.stderr)

            new_time = time.time()
            if new_time - lasttime <= 1.2:
                time.sleep(1.2 - new_time + lasttime)
            lasttime = new_time


if __name__ == '__main__':
    bibmain()
