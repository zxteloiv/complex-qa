from typing import Mapping, List, Generator, Tuple, Union, Optional, Callable
from trialbot.data.translator import FieldAwareTranslator
from trialbot.data.fields import SeqField
from trialbot.data.field import Field, NullableTensor

PREPROCESS_HOOK = Callable[[str], str]  # read, modify, and return a sentence string
PREPROCESS_HOOKS = List[PREPROCESS_HOOK]


class AutoPLMField(Field):
    def batch_tensor_by_key(self, tensors_by_keys: Mapping[str, List[NullableTensor]]) -> Mapping[str, 'torch.Tensor']:
        sent_list = tensors_by_keys.get(self.renamed_key)
        plm_inputs = self.tokenizer(sent_list, padding=self.use_padding, return_tensors="pt")
        plm_inputs = plm_inputs.data    # use the real dict instead of the UserDict
        return {self.renamed_key: plm_inputs}

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        # for PLM, pretrained tokenizers can replace anything and do not require a vocab for the field
        yield from []

    def to_tensor(self, example) -> Mapping[str, Union[NullableTensor, str]]:
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
                 source_preprocess_hooks: Optional[List[Callable[[str], str]]] = None,
                 **auto_tokenizer_kwargs,
                 ):
        super().__init__(field_list=[
            AutoPLMField(source_key=source_field,
                         auto_plm_name=auto_plm_name,
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


