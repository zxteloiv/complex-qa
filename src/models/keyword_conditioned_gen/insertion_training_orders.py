from typing import Callable, List, Generator, Iterable, Union, Tuple
import torch

_DECODER_INPUT = List[torch.LongTensor]
_NUMBER_ARRAY = torch.LongTensor
_PRED_LOC = _NUMBER_ARRAY # a batch of indices to slots
_PRED_TOK = _NUMBER_ARRAY # a batch of indices to vocabulary set

INSERTION_ORDER = Generator[Tuple[_DECODER_INPUT, _PRED_LOC, _PRED_TOK], None, None]

class TrainingOrder:
    """
    A general interface that specifies how an insertion transformer should be trained.
    """
    def __call__(self, *args, **kwargs):
        yield from self.example_order_gen(*args, **kwargs)

    def example_order_gen(self,
                          target_tokens: List[torch.LongTensor]
                          ) -> INSERTION_ORDER:
        raise NotImplemented


class L2ROrder(TrainingOrder):
    def __init__(self, eos_id: int):
        self.eos_id: int = eos_id

    def example_order_gen(self,
                          target_tokens: List[torch.LongTensor]
                          ) -> INSERTION_ORDER:
        """
        Yields decoder training samples from left to right.

        :param target_tokens: [(tgt_len,)]
        :return: Generator tuple of (decoder input, gold tokens, gold locations)
        """
        assert all(x.ndimension() == 1 for x in target_tokens)
        max_len = max(x.size()[0] for x in target_tokens)
        batch_size = len(target_tokens)

        sample_tensor = target_tokens[0]

        for l in range(max_len):
            dec_inp = []
            toks = []
            locs = [l] * batch_size

            for x in target_tokens:
                dec_inp.append(x[:l])
                toks.append(self.eos_id if l >= x.size()[0] else x[l])

            toks = torch.tensor(data=toks, dtype=torch.long, device=sample_tensor.device)
            locs = torch.tensor(data=locs, dtype=torch.long, device=sample_tensor.device)

            # dec_inp: list of torch.LongTensor
            # toks: torch.LongTensor (1d)
            # locs: torch.LongTensor (1d)
            yield dec_inp, toks, locs

