from typing import List, Generator, Tuple, Mapping
from training.trial_bot.data.translator import Translator
from training.trial_bot.data.ns_vocabulary import START_SYMBOL, END_SYMBOL, PADDING_TOKEN

import logging
import torch
from torch.nn.utils.rnn import pad_sequence

from training.trial_bot.trial_registry import Registry

@Registry.translator()
class CharBasedWeiboKeywordsTranslator(Translator):
    def __init__(self, max_len: int = 30):
        super(CharBasedWeiboKeywordsTranslator, self).__init__()
        self.max_len = max_len
        self.shared_namespace = "tokens"

    @staticmethod
    def filter_split_str(sent: str) -> List[str]:
        return list(filter(lambda x: x not in ("", ' ', '\t'), list(sent)))

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        if len(example) != 3:
            logging.warning('Skip invalid line: %s' % str(example))
            return

        src, tgt, kwds = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = [START_SYMBOL] + self.filter_split_str(tgt)[:self.max_len] + [END_SYMBOL]
        kwds = self.filter_split_str("".join(kwds))

        tokens = src + tgt + kwds
        for tok in tokens:
            yield self.shared_namespace, tok

    def to_tensor(self, example):
        assert self.vocab is not None

        src, tgt, kwds = example
        src = self.filter_split_str(src)[:self.max_len]
        tgt = [START_SYMBOL] + self.filter_split_str(tgt)[:self.max_len] + [END_SYMBOL]
        kwds = self.filter_split_str("".join(kwds))

        def tokens_to_id_vector(token_list: List[str]) -> torch.Tensor:
            ids = [self.vocab.get_token_index(tok, self.shared_namespace) for tok in token_list]
            return torch.tensor(ids, dtype=torch.long)

        tensors = map(tokens_to_id_vector, (src, tgt, kwds))
        return dict(zip(("source_tokens", "target_tokens", "keyword_tokens"), tensors))

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]):
        assert len(tensors) > 0
        keys = tensors[0].keys()

        tensor_list_by_keys = dict((k, []) for k in keys)
        for instance in tensors:
            for k, tensor in instance.items():
                assert tensor.ndimension() == 1
                tensor_list_by_keys[k].append(tensor)

        batched_tensor = dict(
            (
                k,
                pad_sequence(tlist, batch_first=True,
                             padding_value=self.vocab.get_token_index(PADDING_TOKEN, self.shared_namespace))
            )
            for k, tlist in tensor_list_by_keys.items()
        )

        return batched_tensor

