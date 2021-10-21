from typing import List, Mapping, Generator, Tuple, Optional, Any, Literal, Union
from trialbot.data.field import Field, NullableTensor
import torch
from trialbot.data import START_SYMBOL, END_SYMBOL, PADDING_TOKEN, NSVocabulary
from itertools import product
from utils.preprocessing import nested_list_numbers_to_tensors

class CharArrayField(Field):
    def get_sent(self, example):
        return example.get(self.source_key)

    def tear_to_char_array(self, raw_data: Union[None, str, Any]) -> List[List[str]]:
        if not isinstance(raw_data, str):
            return []
        # empty string "   " will give an empty arr as []
        arr = [list(word.lower()) if self.lower_case else list(word) for word in raw_data.strip().split()]
        return arr

    def array_to_id(self, char_arr: List[List[str]]) -> List[List[int]]:
        char_id_arr = [[self.vocab.get_token_index(ch, self.ns) for ch in char_list] for char_list in char_arr]
        return char_id_arr

    def clip_array(self, char_arr: List[List[Union[int, str]]]) -> List[List[Union[int, str]]]:
        """Clip the char array of an example, NOT applicable to a batch"""
        clipped_arr = char_arr
        if self.max_word_seq_len > 0:
            clipped_arr = char_arr[:self.max_word_seq_len]

        if self.max_char_seq_len > 0:
            for i, word in enumerate(clipped_arr):
                clipped_arr[i] = word[:self.max_char_seq_len]
        return clipped_arr

    def generate_namespace_tokens(self, example) -> Generator[Tuple[str, str], None, None]:
        if self.add_start_end_toks:
            yield from product([self.ns], [START_SYMBOL, END_SYMBOL])

        seq_raw = self.get_sent(example)
        char_arr = self.tear_to_char_array(seq_raw)
        clipped_arr = self.clip_array(char_arr)
        for word in clipped_arr:
            for ch in word:
                yield self.ns, ch

    def to_tensor(self, example) -> Mapping[str, Union[NullableTensor, List[Any]]]:
        seq_raw = self.get_sent(example)
        char_arr = self.tear_to_char_array(seq_raw)
        if char_arr is None or len(char_arr) == 0:  # any error occurred during splitting into chars
            return {self.renamed_key: None}

        clipped_arr = self.clip_array(char_arr)
        if self.add_start_end_toks:
            for i, w in enumerate(clipped_arr):
                clipped_arr[i] = [START_SYMBOL] + w + [END_SYMBOL]

        id_arr = self.array_to_id(clipped_arr)
        return {self.renamed_key: id_arr}

    def batch_tensor_by_key(self,
                            tensors_by_keys: Mapping[str, Union[NullableTensor, List[Any]]]
                            ) -> Mapping[str, torch.Tensor]:
        tensor_list = tensors_by_keys.get(self.renamed_key)
        if tensor_list is None or len(tensor_list) == 0:
            raise KeyError(f'Empty field key {self.renamed_key} confronted. Failed to build the instance tensors')

        batch_tensor = nested_list_numbers_to_tensors(tensor_list, self.padding)
        return {self.renamed_key: batch_tensor}

    def __init__(self, source_key: str,
                 renamed_key: str = None,
                 namespace: str = None,
                 add_start_end_toks: bool = True,
                 padding_id: int = 0,
                 max_word_seq_len: int = 0,
                 max_char_seq_len: int = 0,
                 use_lower_case: bool = True,
                 ):
        super().__init__()
        self.source_key = source_key
        self.ns = namespace or source_key
        self.renamed_key = renamed_key or source_key
        self.add_start_end_toks = add_start_end_toks
        self.padding = padding_id
        self.max_word_seq_len = max_word_seq_len
        self.max_char_seq_len = max_char_seq_len
        self.lower_case = use_lower_case
