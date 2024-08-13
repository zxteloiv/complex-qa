import json


def main(filename: str):
    for i, l in enumerate(open(filename)):
        data = json.loads(l.strip())
        query = data['title']
        for e in data['elist']:
            if not filter_rule(query, e):
                continue

            print_s2entry(e, query)


def print_s2entry(e: dict, query=None):
    s2id = e['paperId']
    corpus_id = e['corpusId']
    doi = e['externalIds'].get('DOI', None)
    title = e['title']
    venue = e['venue']
    year = e['year']
    cits = e['citationCount']
    tldr = e.get('tldr')
    if tldr is not None:
        tldr = tldr.get('text')
    authors = ','.join(x['name'] for x in e['authors'])
    abstract = e['abstract']
    if abstract is not None:
        abstract = abstract.replace('\t', ' ').replace('\n', ' ')

    print('\t'.join(str(x) for x in [
        s2id, corpus_id, doi, cits,
        query, title, year, venue, authors,
        tldr, abstract
    ]))


def filter_rule(query: str, e: dict) -> bool:
    title = e['title']
    year = e['year'] or 0
    cits = e['citationCount']
    tldr = e.get('tldr')
    if tldr is not None:
        tldr = tldr.get('text')
    abstract = e['abstract']
    if abstract is not None:
        abstract = abstract.replace('\t', ' ').replace('\n', ' ')

    # rules
    if year < 2020:
        return False

    if year < 2023 and cits <= 0:
        return False

    txt_fields = (title, query, tldr, abstract)

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

    if (year < 2023 and cits >= 5) or (year >= 2023 and cits >= 2):
        return True

    return False


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
