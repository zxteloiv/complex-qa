from itertools import product
from typing import Any
from collections.abc import Iterator

import torch
from trialbot.data.translator import Field, FieldAwareTranslator, T

from utils.preprocessing import nested_list_numbers_to_tensors
from .squall_translator import NLTableField

# copied from WikiSQL repo: lib/query.py
WIKISQL_AGG_OPS = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
WIKISQL_COND_OPS = ['=', '>', '<', 'OP']


class WikiSQLSrcField(NLTableField):
    def get_nl_words_list(self, example: dict):
        return self._tokenizer.tokenize(example['question'])

    def get_col_words_list(self, example: dict):
        return example['table']['header']


class SQLovaDecodingField(Field):
    def __init__(self, plm_name: str):
        super().__init__()
        self.padding = -1
        self.pad_keys = ['select', 'agg', 'num_headers', 'num_conds',
                         'where_num', 'where_cols', 'where_ops', 'where_begin', 'where_end']
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(plm_name)

    def build_batch_by_key(self, input_dict: dict[str, list[T]]) -> dict[str, torch.Tensor | list[T]]:
        output = {k: nested_list_numbers_to_tensors(input_dict[k], self.padding) for k in self.pad_keys}
        output['cond_values'] = input_dict['cond_values']
        return output

    def generate_namespace_tokens(self, example: Any) -> Iterator[tuple[str, str]]:
        yield from []   # no need to define

    def to_input(self, example) -> dict[str, T | None]:
        sql_dict = example['sql']

        table = example['table']
        n_cols = len(table['header'])
        conds = sql_dict['conds']

        output = {
            'select': sql_dict['sel'],  # (b,), no mask
            'agg': sql_dict['agg'],     # (b,), no mask
            'where_num': len(conds),    # (b,), no mask
            'num_headers': n_cols,      # (b,), no mask
            'num_conds': len(conds),    # (b,), no mask
        }

        where_cols = [0] * n_cols
        where_ops = [-1] * n_cols
        where_begin = [-1] * n_cols
        where_end = [-1] * n_cols
        cond_val = []
        nl_words: list[str] = self._tokenizer.tokenize(example['question'])

        for col, op, span in conds:
            span = str(span)
            where_cols[col] = 1
            where_ops[col] = op
            cond_val.append(span)

            span_words: list[str] = self._tokenizer.tokenize(span)
            first_word, last_word = span_words[0], span_words[-1]
            if first_word not in nl_words:
                where_begin = None
                break
            begin = nl_words.index(first_word)

            if last_word not in nl_words[begin:]:
                where_end = None
                break
            end = begin if len(span_words) == 1 else nl_words.index(last_word, begin)

            where_begin[col] = begin
            where_end[col] = end

        output['where_cols'] = where_cols   # (b, n_col), self-padded by -1, valid values are 0 and 1
        output['where_ops'] = where_ops     # (b, n_col), self-padded by -1, valid values are WIKISQL_COND_OPS indices
        output['cond_values'] = cond_val    # a list of string lists, where each string is the copied value

        output['where_begin'] = where_begin  # (b, n_col), self-padded by -1, values are word indices
        output['where_end'] = where_end      # (b, n_col), self-padded by -1, values are word indices
        return output


class SQLovaTranslator(FieldAwareTranslator):
    def __init__(self, plm_name: str = 'bert-base-uncased'):
        super().__init__(field_list=[
            WikiSQLSrcField(plm_name),
            SQLovaDecodingField(plm_name),
        ])


def install_translators(reg=None):
    if reg is None:
        from trialbot.training import Registry
        reg = Registry._translators

    reg['sqlova'] = SQLovaTranslator