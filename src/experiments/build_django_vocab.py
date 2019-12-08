
import sys
sys.path = ['..'] + sys.path
from typing import Dict
from tqdm import tqdm
from datasets import django_rank, django_rank_translator
from trialbot.data import NSVocabulary

def main():
    translator = django_rank_translator.DjangoRankTranslator(70)
    train_data, _, _ = django_rank.django_pure_none()

    counter: Dict[str, Dict[str, int]] = dict()
    for example in tqdm(iter(train_data)):
        for namespace, w in translator.generate_namespace_tokens(example):
            if namespace not in counter:
                counter[namespace] = dict()

            if w not in counter[namespace]:
                counter[namespace][w] = 0

            counter[namespace][w] += 1

    # same as the TranX settings.
    print(f'counter basic stats: nl={len(counter["nl"])}, lf={len(counter["lf"])}')
    vocab = NSVocabulary(counter,
                         min_count=({"nl": 15, "lf": 15}),
                         max_vocab_size=5000,
                         )
    print('saving to ./django_vocab_15')
    vocab.save_to_files('./django_vocab_15')
    print(vocab)

if __name__ == '__main__':
    main()
