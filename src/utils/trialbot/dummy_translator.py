from functools import partial
from typing import Mapping, Generator, Tuple, List

from trialbot.data.translator import Translator, NullableTensor, FieldAwareTranslator


class DummyTranslator(Translator):
    def batch_tensor(self, tensors: List[Mapping[str, NullableTensor]]):
        if self.filter_none:
            tensors = list(filter(lambda x: all(v is not None for v in x.values()), tensors))
        batch_by_fields = FieldAwareTranslator.list_of_dict_to_dict_of_list(tensors)
        if self.return_tuples and self.gather_keys is not None:
            return tuple(batch_by_fields[k] for k in self.gather_keys)
        else:
            return batch_by_fields

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        yield from []

    def to_tensor(self, example) -> Mapping[str, NullableTensor]:
        return example if self.gather_keys is None else {k: example.get(k) for k in self.gather_keys}

    def __init__(self, filter_none: bool = False, gather_keys: list = None, return_tuples: bool = False):
        super().__init__()
        self.filter_none = filter_none
        self.return_tuples = return_tuples
        self.gather_keys = gather_keys


def install_dummy_translator(reg: dict = None,
                             name: str = 'dummy',
                             filter_none: bool = False,
                             gather_keys: list = None,
                             return_tuples: bool = False,
                             ) -> None:
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._translators

    reg[name] = partial(DummyTranslator, filter_none=filter_none, gather_keys=gather_keys,
                        return_tuples=return_tuples)

