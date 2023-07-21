from typing import Optional, Callable
from trialbot.data.translator import FieldAwareTranslator
from trialbot.data.fields import SeqField
from trialbot.data.field import Field, T
from typing import Any
import torch
from collections.abc import Iterator

PREPROCESS_HOOK = Callable[[str], str]  # read, modify, and return a sentence string
PREPROCESS_HOOKS = list[PREPROCESS_HOOK]


class AutoPLMField(Field):
    def build_batch_by_key(self, input_dict: dict[str, list[T]]) -> dict[str, torch.Tensor | list[T]]:
        sent_list = input_dict.get(self.renamed_key)
        plm_inputs = self.tokenizer(sent_list, padding=self.use_padding, return_tensors="pt")
        plm_inputs = plm_inputs.data    # use the real dict instead of the UserDict
        return {self.renamed_key: plm_inputs}

    def generate_namespace_tokens(self, example: Any) -> Iterator[tuple[str, str]]:
        # for PLM, pretrained tokenizers can replace anything and do not require a vocab for the field
        yield from []

    def to_input(self, example) -> dict[str, T | None]:
        sent = example.get(self.source_key)
        if sent is None:
            return {self.renamed_key: None}

        for hook in self.preprocess_hooks:
            sent = hook(sent)

        return {self.renamed_key: sent}

    def __init__(self,
                 source_key: str,
                 auto_plm_name: str = 'bert-base-uncased',
                 renamed_key: str = 'model_inputs',
                 use_padding: bool = True,
                 preprocess_hooks: Optional[PREPROCESS_HOOKS] = None,
                 **auto_tokenizer_kwargs,
                 ):
        super().__init__()
        from transformers import AutoTokenizer
        self.source_key = source_key
        self.renamed_key = renamed_key
        self.tokenizer = AutoTokenizer.from_pretrained(auto_plm_name, **auto_tokenizer_kwargs)
        self.use_padding = use_padding
        self.preprocess_hooks = preprocess_hooks or []


class PLM2SeqTranslator(FieldAwareTranslator):
    """Predefined translator for common use.
    If the desired customization is not available, write a new translator instead."""
    def __init__(self,
                 auto_plm_name: str,
                 source_field: str,
                 target_field: str,
                 target_max_token: int = 0,
                 use_lower_case: bool = True,
                 source_preprocess_hooks: Optional[list[Callable[[str], str]]] = None,
                 **auto_tokenizer_kwargs,
                 ):
        super().__init__(field_list=[
            AutoPLMField(source_key=source_field,
                         auto_plm_name=auto_plm_name,
                         renamed_key="source_tokens",
                         preprocess_hooks=source_preprocess_hooks,
                         **auto_tokenizer_kwargs,
                         ),
            SeqField(source_key=target_field,
                     renamed_key="target_tokens",
                     namespace="target_tokens",
                     max_seq_len=target_max_token,
                     use_lower_case=use_lower_case
                     ),
        ])


