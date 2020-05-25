
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

def atis():
    from datasets import atis_rank, atis_rank_translator
    translator = atis_rank_translator.AtisRankTranslator(70)
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

def django():
    from datasets import django_rank, django_rank_translator
    translator = django_rank_translator.DjangoRankTranslator(70)
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
    import sys
    dataset = sys.argv[1]
    if dataset == "atis":
        atis()

    elif dataset == "django":
        django()

    else:
        print("dataset not available")
