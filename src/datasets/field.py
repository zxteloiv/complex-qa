from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal
from trialbot.data import Translator, NSVocabulary
from collections import defaultdict
import torch

class Field(Translator):
    """
    A field is not actually a translator, this is experimental
    and will later be refactored to inherit from another interface.
    TODO: remove Field dependency with Translator.
    """
    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        """
        A normal translator accept a batch of tensors,
        which is an iterable collection containing the instances from Translator.to_tensor interface.
        A field will handle the tensor list by some predefined key,
        but will accept all the possible tensors batch dict by the keys.
        :param tensors_by_keys:
        :return:
        """
        raise NotImplementedError


class FieldAwareTranslator(Translator):
    def __init__(self, field_list: List[Field]):
        super().__init__()
        self.fields = field_list

    def index_with_vocab(self, vocab: NSVocabulary):
        super().index_with_vocab(vocab)
        for field in self.fields:
            field.index_with_vocab(vocab)

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        for field in self.fields:
            yield from field.generate_namespace_tokens(example)

    def to_tensor(self, example) -> Mapping[str, torch.Tensor]:
        instance_fields = {}
        for field in self.fields:
            instance_fields.update(field.to_tensor(example))
        return instance_fields

    def batch_tensor(self, tensors: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
        batch_dict = self.list_of_dict_to_dict_of_list(tensors)
        output = {}
        for field in self.fields:
            output.update(field.batch_tensor_by_key(batch_dict))
        return output

    @staticmethod
    def list_of_dict_to_dict_of_list(ld: List[Mapping[str, Any]]) -> Mapping[str, List[Any]]:
        list_by_keys = defaultdict(list)
        for d in ld:
            for k, v in d.items():
                list_by_keys[k].append(v)
        return list_by_keys


