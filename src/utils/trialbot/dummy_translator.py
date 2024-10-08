from functools import partial
from operator import itemgetter
from typing import Any, cast

from trialbot.data.translator import FieldAwareTranslator, Field, T
from collections.abc import Iterator
import torch


class DummyTranslator(FieldAwareTranslator):
    def __init__(self, filter_none: bool = False, gather_keys: list = None, renamed_to: list = None):
        super().__init__(field_list=[DummyField(gather_keys, renamed_to)], filter_none=filter_none)
        self.gather_keys = gather_keys

    def lazy_init(self, gather_keys: list[str] | None = None, renamed_to: list[str] | None = None):
        field: DummyField = cast(DummyField, self.fields[0])
        field.gather_keys = gather_keys
        field.renamed_to = renamed_to

    def to_tuple(self, example) -> tuple:
        d = self.to_input(example)
        field: DummyField = cast(DummyField, self.fields[0])
        return itemgetter(*field.gather_keys)(d)


class DummyField(Field):
    def build_batch_by_key(self, input_dict: dict[str, list[T]]) -> dict[str, torch.Tensor | list[T]]:
        if self.gather_keys is None:
            return input_dict
        elif self.renamed_to is None:
            return {k: input_dict.get(k) for k in self.gather_keys}
        else:
            return {k: input_dict.get(k) for k in self.renamed_to}

    def generate_namespace_tokens(self, example: Any) -> Iterator[tuple[str, str]]:
        yield from []

    def to_input(self, example) -> dict[str, T | None]:
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

