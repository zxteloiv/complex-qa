from typing import List, Any, Dict
from trialbot.data.translator import Translator
from tqdm import tqdm


def get_ns_counter(train_data: List[Any], translator: Translator) -> Dict[str, Dict[str, int]]:
    counter: Dict[str, Dict[str, int]] = dict()
    for example in tqdm(train_data):
        for namespace, w in translator.generate_namespace_tokens(example):
            if namespace not in counter:
                counter[namespace] = dict()

            if w not in counter[namespace]:
                counter[namespace][w] = 0

            counter[namespace][w] += 1
    return counter

