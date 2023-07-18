from functools import partial
from typing import Mapping, Generator, Tuple, List

from trialbot.data.translator import Translator, NullableTensor, FieldAwareTranslator, Field


class DummyTranslator(FieldAwareTranslator):
    def __init__(self, filter_none: bool = False, gather_keys: list = None):
        super().__init__(field_list=[DummyField(gather_keys)], filter_none=filter_none)


class DummyField(Field):
    def batch_tensor_by_key(self, tensors_by_keys):
        return tensors_by_keys

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        yield from []

    def to_tensor(self, example) -> Mapping[str, NullableTensor]:
        if self.gather_keys is None:
            return example

        elif self.renamed_to is None:
            return {k: example.get(k) for k in self.gather_keys}

        else:
            return {k_: example.get(k) for k, k_ in zip(self.gather_keys, self.renamed_to)}

    def __init__(self, gather_keys: list[str] | None = None, renamed_to: list[str] | None = None):
        super().__init__()
        self.gather_keys = gather_keys
        self.renamed_to = renamed_to


def install_dummy_translator(reg: dict = None,
                             name: str = 'dummy',
                             filter_none: bool = False,
                             gather_keys: list[str] | None = None,
                             ) -> None:
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._translators

    reg[name] = partial(DummyTranslator, filter_none=filter_none, gather_keys=gather_keys)

