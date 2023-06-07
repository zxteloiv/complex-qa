from trialbot.data.field import Field
from trialbot.data.translator import Translator, NullableTensor, FieldAwareTranslator


class DummyField(Field):
    def batch_tensor_by_key(self, tensors_by_keys):
        # nothing is translated to tensors, all kept in the raw python data types
        return tensors_by_keys

    def to_tensor(self, example):
        return example

    def generate_namespace_tokens(self, example):
        yield from []


class DummyTranslator(FieldAwareTranslator):
    def __init__(self, filter_none: bool = False):
        super().__init__(field_list=[DummyField()], vocab_fields=[], filter_none=filter_none)


def install_dummy_translator(reg: dict = None, name: str = 'dummy') -> None:
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._translators

    reg[name] = DummyTranslator

