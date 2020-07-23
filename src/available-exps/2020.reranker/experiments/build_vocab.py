
import sys
sys.path = ['..'] + sys.path
from typing import Dict
from tqdm import tqdm
from trialbot.data import NSVocabulary

def _get_counter(train_data, translator):
    counter: Dict[str, Dict[str, int]] = dict()
    for example in tqdm(iter(train_data)):
        for namespace, w in translator.generate_namespace_tokens(example):
            if namespace not in counter:
                counter[namespace] = dict()

            if w not in counter[namespace]:
                counter[namespace][w] = 0

            counter[namespace][w] += 1
    return counter

def atis(g):
    from datasets import atis_rank, atis_rank_translator
    if g == 'word':
        translator = atis_rank_translator.AtisRankTranslator(70)
    else:
        translator = atis_rank_translator.AtisRankChTranslator(70)
    train_data, _, _ = atis_rank.atis_pure_none()

    counter = _get_counter(train_data, translator)

    # same as the TranX settings.
    print('counter basic stats: ' + ",".join(f"{ns}={len(counts)}" for ns, counts in counter.items()))
    vocab = NSVocabulary(counter,
                         min_count=({"nl": 0, "lf": 0, "nlch": 0, "lfch": 0}),
                         max_vocab_size=5000,
                         )
    print('saving to ./atis_vocab')
    vocab.save_to_files('./atis_vocab')
    print(vocab)

def django(g):
    from datasets import django_rank, django_rank_translator
    if g == 'word':
        translator = django_rank_translator.DjangoRankTranslator(70)
    else:
        translator = django_rank_translator.DjangoRankChTranslator(70)
    train_data, _, _ = django_rank.django_pure_none()

    counter = _get_counter(train_data, translator)

    # same as the TranX settings.
    print('counter basic stats: ' + ",".join(f"{ns}={len(counts)}" for ns, counts in counter.items()))
    vocab = NSVocabulary(counter,
                         min_count=({"nl": 15, "lf": 15, "nlch": 0, "lfch": 0}),
                         max_vocab_size=5000,
                         )
    print('saving to ./django_vocab_15')
    vocab.save_to_files('./django_vocab_15')
    print(vocab)

if __name__ == '__main__':
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', choices=["atis", "django"])
    parser.add_argument('--granularity', '-g', choices=['word', 'char'], default='word')
    args = parser.parse_args()

    if args.dataset == 'atis':
        atis(args.granularity)

    elif args.dataset == "django":
        django(args.granularity)

    else:
        print("dataset not available")
