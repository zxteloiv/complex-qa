import sys
from collections import defaultdict
from itertools import product
from typing import Sequence, Generator

import tqdm
from trialbot.utils.root_finder import find_root

sys.path.insert(0, find_root('.SRC'))


def iterngrams(s: Sequence, n: int, start=None, end=None) -> Generator[Sequence, None, None]:
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f'n-gram requires a positive n, but got {n}')

    start = start or 0
    end = end or len(s)
    for i in range(start, end):
        if i + n <= end:
            yield s[i:i+n]
        else:
            break


def get_all_ngrams(s: Sequence, n_min=1, n_max=None):
    n_max = n_max or len(s)
    for n in range(n_min, n_max + 1):
        yield from iterngrams(s, n)


def main(args):
    from shujuji.comp_gen_bundle import install_raw_qa_datasets
    from trialbot.training import Registry
    install_raw_qa_datasets(Registry._datasets)
    train, dev, test = Registry.get_dataset(args.dataset)
    print(len(train), len(dev), len(test))

    nl_max, lf_max = 1, 1
    pmi, pair_freq, nl_freq, lf_freq = get_pmi_for_dataset(train, nl_max, lf_max)
    output_file = open(f'pmi-nl{nl_max}-lf{lf_max}.txt', 'w')
    for (nl, lf), v in sorted(pmi.items(), key=lambda t: t[1], reverse=True):
        output_file.write(f"{v}\t{nl_freq[nl]}\t{lf_freq[lf]}\t{nl}\t{lf}\n")
    output_file.close()


def get_pmi_for_dataset(dataset, nl_max, lf_max):
    pair_freq = defaultdict(lambda: 0)
    nl_freq = defaultdict(lambda: 0)
    lf_freq = defaultdict(lambda: 0)

    for x in tqdm.tqdm(dataset):
        sent, sql = x['sent'], x['sql']
        nl_ngrams = [tuple(ngram) for ngram in get_all_ngrams(sent.split(), n_max=nl_max)] + [()]
        lf_ngrams = [tuple(ngram) for ngram in get_all_ngrams(sql.split(), n_max=lf_max)] + [()]
        for ngram in nl_ngrams:
            nl_freq[ngram] += 1
        for ngram in lf_ngrams:
            lf_freq[ngram] += 1
        for nl_ngram, lf_ngram in product(nl_ngrams, lf_ngrams):
            pair_freq[nl_ngram, lf_ngram] += 1

    pmi = defaultdict(lambda: 0.)
    for nl, lf in tqdm.tqdm(product(nl_freq.keys(), lf_freq.keys()), total=len(nl_freq) * len(lf_freq)):
        if (nl, lf) in pair_freq:
            pmi[nl, lf] = pair_freq[nl, lf] / nl_freq[nl] / lf_freq[lf]

    return pmi, pair_freq, nl_freq, lf_freq


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='raw_qa.scholar_cg')
    main(parser.parse_args())
