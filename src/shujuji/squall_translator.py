import logging
from collections import defaultdict
from enum import IntEnum
from typing import Generator, Tuple, List, Optional, Set, Dict
from typing import Any
from collections.abc import Iterator

import torch
from trialbot.data.translator import Field, FieldAwareTranslator, T
from trialbot.training import Registry
from trialbot.data import NSVocabulary, START_SYMBOL, END_SYMBOL

from utils.graph import Graph, dfs_walk
from utils.preprocessing import nested_list_numbers_to_tensors


@Registry.translator('squall-base')
class SquallTranslator(FieldAwareTranslator):
    def __init__(self, plm_model: str = 'bert-base-uncased', ns_keyword: str = 'keyword', ns_coltype: str = 'col_type'):
        super().__init__(
            field_list=[
                NLTableField(plm_model),
                SquallAllInOneField(ns_keyword, ns_coltype),
            ]
        )


class SrcType(IntEnum):
    Padding = 0
    # since no special tokens are utilized by the decoder in the squall baseline,
    # it's not necessary to differentiate special tokens from words' side or columns' side.
    Special = 1
    Word = 2
    WordPivot = 3
    Column = 4
    ColPivot = 5


class TgtType(IntEnum):
    Padding = 0
    Keyword = 1
    Column = 2
    LiteralString = 3
    LiteralNumber = 4


class NLTableField(Field):
    def build_batch_by_key(self, input_dict: dict[str, list[T]]) -> dict[str, torch.Tensor | list[T]]:
        output = {}
        for pad_id, keys in self.padded_keys.items():
            for key in keys:
                ids = input_dict[key]
                output[key] = nested_list_numbers_to_tensors(ids, padding=pad_id)
        return output

    def generate_namespace_tokens(self, example: Any) -> Iterator[tuple[str, str]]:
        yield from []   # nothing to vocab, because the pretrained-lm knows all

    def to_input(self, example) -> dict[str, T | None]:
        src_params, word_locs, col_locs = self._get_source(example)
        self._word_locs = word_locs
        self._col_locs = col_locs
        return src_params

    def get_loc_mappings(self):
        return self._word_locs, self._col_locs

    def get_nl_words_list(self, example):
        return example['nl']

    def get_col_words_list(self, example):
        return [col[0] for col in example['columns']]

    def _get_source(self, example) -> Tuple[dict, list, list]:
        tokens, types = [], []

        def _add(tok, toktype):
            tokens.append(tok)
            types.append(toktype)

        def _add_nl_tokens(nl):
            physical_loc = []
            _add('[CLS]', SrcType.Special)

            for word in nl:
                # the dataset doesn't have empty word tokens.
                for piece in self._tokenizer.tokenize(word):
                    _add(piece, SrcType.Word)
                types[-1] = SrcType.WordPivot   # the last piece is selected to represent the world
                physical_loc.append(len(tokens) - 1)

            _add('[SEP]', SrcType.Special)
            return physical_loc

        def _add_col_tokens(cols):
            physical_loc = []
            for col in cols:
                pieces = self._tokenizer.tokenize(col)
                for piece in pieces:
                    _add(piece, SrcType.Column)

                if len(pieces) == 0:    # empty column label, in case that the alignment index is corrupted
                    _add('.', SrcType.ColPivot)
                else:
                    types[-1] = SrcType.ColPivot

                physical_loc.append(len(tokens) - 1)
                _add('[SEP]', SrcType.Special)

            return physical_loc

        # indicating the pivot index (physical location) w.r.t each word index (virtual location)
        word_loc_lookup = _add_nl_tokens(self.get_nl_words_list(example))
        plm_type_ids = [0] * len(tokens)    # 0 for question tokens, 1 for schema tokens in BERT

        # indicating the pivot index (physical location) w.r.t each column index (virtual location)
        col_loc_lookup = _add_col_tokens(self.get_col_words_list(example))
        plm_type_ids.extend([1] * (len(tokens) - len(plm_type_ids)))

        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        params = {
            "src_ids": token_ids,
            "src_types": types,
            "src_plm_type_ids": plm_type_ids,
        }
        return params, word_loc_lookup, col_loc_lookup

    def __init__(self, plm_name: str = 'bert-base-uncased',):
        super().__init__()
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(plm_name)
        self.padded_keys: Dict[int, List[str]] = {
            SrcType.Padding: ['src_ids', 'src_types', 'src_plm_type_ids'],
        }


class SquallAllInOneField(Field):
    def build_batch_by_key(self, input_dict: dict[str, list[T]]) -> dict[str, torch.Tensor | list[T]]:
        output = {}
        for pad_id, keys in self.padded_keys.items():
            for key in keys:
                ids = input_dict[key]
                output[key] = nested_list_numbers_to_tensors(ids, padding=pad_id)

        output['tbl_cells'] = input_dict['tbl_cells']  # a batch list of cell text sets
        output['nl_toks'] = input_dict['nl_toks']
        output['sql_toks'] = input_dict['sql_toks']
        return output

    def generate_namespace_tokens(self, example: Any) -> Iterator[tuple[str, str]]:
        # other supervisions are all attention-based and thus dynamic.
        # the token types are enumerable as the TgtType class above.
        # Therefore, only the keywords and column types require a predefined vocabulary for predictions.
        sql = example['sql']
        yield self.ns_keyword, START_SYMBOL
        yield self.ns_keyword, END_SYMBOL
        for tok_type, value, span in sql:
            if tok_type == "Keyword":
                yield self.ns_keyword, value
        for col in example['columns']:
            for col_type in col[2]:
                yield self.ns_coltype, col_type
        yield self.ns_coltype, 'none'

    def to_input(self, example) -> dict[str, T | None]:
        alignments = self._get_attn_sup(example)
        tgt_params = self._get_target(example)
        col_type_mask = self._get_col_type_mask(example)
        table_cells = self._load_text_cells(example)

        id_lists = dict()
        id_lists.update(col_type_mask)
        id_lists.update(alignments)
        id_lists.update(tgt_params)
        id_lists['tbl_cells'] = table_cells
        id_lists['nl_toks'] = example.get('nl')
        id_lists['sql_toks'] = [x[1] for x in example.get('sql')]
        return id_lists

    def _get_col_type_mask(self, example) -> dict:
        # in fact, the number of col-types is less than 100,
        # and it is believed not large across different tasks.
        # we can reliably represent the candidate with one-hot masks for the ease of use,
        # instead of specifying the selected type ids.
        col_type_candidates = []
        for i, col in enumerate(example['columns']):
            type_ids = [self.vocab.get_token_index(coltype, self.ns_coltype) for coltype in col[2]]
            if len(type_ids) == 0:
                # any column has a type, if no parsed type is available, the none is used.
                # indicating the column copy module must select a "none" type.
                type_ids.append(self.vocab.get_token_index('none', self.ns_coltype))

            one_hot_mask = [1 if i in type_ids else 0 for i in range(self.vocab.get_vocab_size(self.ns_coltype))]
            col_type_candidates.append(one_hot_mask)

        return {"col_type_mask": col_type_candidates}

    def _load_text_cells(self, example) -> Set[Tuple[str, str]]:
        table = example['tbl_cells']
        ret = set()     # each element is col_str, cell_str
        for content in table["contents"][2:]:
            for col in content:
                if col["type"] == "TEXT":
                    for x in col["data"]:
                        ret.add((col["col"], str(x)))
                elif col["type"] == "LIST TEXT":
                    for lst in col["data"]:
                        for x in lst:
                            ret.add((col["col"], str(x)))
        return ret

    def _get_attn_sup(self, example) -> dict:
        wsc_graph: Graph[str] = self.get_connectivity_graph(example)    # word, sql, column nodes
        num_w, num_c, num_s = map(lambda k: len(example[k]), ('nl', 'columns', 'sql'))
        ws, wc, sc = set(), set(), set()

        # the graph is bidirectional, thus traversal from one kind to another is enough
        for w in range(num_w):
            for node in dfs_walk(wsc_graph, f'word-{w}'):
                ntype, nid = node.split('-')
                nid = int(nid)
                if ntype == 'col':
                    wc.add((w, nid))
                elif ntype == 'sql':
                    ws.add((w, nid))

        for s in range(num_s):
            for node in dfs_walk(wsc_graph, f'sql-{s}'):
                ntype, nid = node.split('-')
                nid = int(nid)
                if ntype == 'col':
                    sc.add((s, nid))

        ws_word = ws_sql = [-1]
        wc_word = wc_col = [-1]
        sc_sql = sc_col = [-1]

        if len(ws) > 0:
            ws_word, ws_sql = map(list, zip(*ws))

        if len(wc) > 0:
            wc_word, wc_col = map(list, zip(*wc))

        if len(sc) > 0:
            sc_sql, sc_col = map(list, zip(*sc))

        alignments = {
            "align_ws_word": ws_word,
            "align_ws_sql": ws_sql,
            "align_wc_word": wc_word,
            "align_wc_col": wc_col,
            "align_sc_sql": sc_sql,
            "align_sc_col": sc_col,
        }
        return alignments

    def _get_target(self, example) -> dict:
        sql = example['sql']
        tgt = defaultdict(list)

        # append start and end token for decoding
        sql = [["Keyword", START_SYMBOL, []]] + sql + [["Keyword", END_SYMBOL, []]]

        for type_str, value, span in sql:
            # although 0 has been reserved for padding, try building mask with tgt_type still
            # because 0 may have meanings in some field like the column=0
            step = {"type": TgtType.Padding, "keyword": 0, "col_id": 0, "col_type": 0,
                    "literal_begin": 0, "literal_end": 0}
            if type_str == "Keyword":
                step['type'] = TgtType.Keyword
                step['keyword'] = self.vocab.get_token_index(value, self.ns_keyword)
            elif type_str == "Column":
                step['type'] = TgtType.Column
                col_id, col_suffix = self._parse_column(value)
                step['col_id'] = col_id
                # take the None col_suffix as "none" (e.g. column c5 as c5_none)
                # must be preprocessed before sent to executors.
                step['col_type'] = self.vocab.get_token_index(col_suffix if col_suffix else "none", self.ns_coltype)
            elif type_str == "Literal.Number":
                step['type'] = TgtType.LiteralNumber
                step['literal_begin'] = span[0]
            elif type_str == "Literal.String":
                # only literal.string has an end
                step['type'] = TgtType.LiteralString
                step['literal_begin'] = span[0]
                step['literal_end'] = span[1]
            else:
                raise ValueError

            # add the tgt_ prefix for better readabilities of the model parameters
            for k, v in step.items():
                tgt['tgt_' + k].append(v)

        return tgt

    @classmethod
    def _parse_column(cls, col: str) -> Tuple[int, Optional[str]]:
        # format: c5_suffix_suffix, or c5 (no suffix)
        upos = col.find('_')
        if upos > 0:
            col_id = int(col[1:upos]) - 1
            col_suffix = col[upos+1:]
        else:
            col_id = int(col[1:]) - 1
            col_suffix = None
        return col_id, col_suffix

    def get_connectivity_graph(self, example) -> Graph[str]:
        nl, cols, sql, nl_ralign, align = map(example.get, ('nl', 'columns', 'sql', 'nl_ralign', 'align'))
        wsc_graph: Graph[str] = Graph()

        def _add_nodes(container, node_type: str):
            for i in range(len(container)):
                wsc_graph.add_v(f'{node_type}-{i}')

        _add_nodes(nl, 'word')
        _add_nodes(cols, 'col')
        _add_nodes(sql, 'sql')

        self._add_w2c_edges(wsc_graph, nl_ralign)
        self._add_sql_edges(wsc_graph, sql)
        self._add_w2s_edges(wsc_graph, align)
        return wsc_graph

    @classmethod
    def _add_w2c_edges(cls, wsc_graph: Graph[str], nl_ralign):
        # nl_ralign format: one list for one word,
        # ["None", null], ["Keyword", ["diff", ""]], ["Column", "c5_number"], ["Literal", "null]
        # only the Column is used, following the baseline

        for i, (align_type, align_val) in enumerate(nl_ralign):
            if align_type == "Column":
                try:
                    col_id, _ = cls._parse_column(align_val)
                except:
                    logging.getLogger(cls.__name__).warning(f'unrecognized column {align_val}')
                    continue
                wsc_graph.add_e_both(f'word-{i}', f'col-{col_id}')

    @classmethod
    def _add_sql_edges(cls, wsc_graph: Graph[str], sql):
        # the sql format is as follows, one list for an SQL token
        #     ["Keyword", "where", []], ["Column", "c1_number", []],
        #     ["Keyword", "=", []], ["Literal.Number", "1", [8]],

        for i, (tok_type, tok_val, tok_span) in enumerate(sql):
            sql_node = f"sql-{i}"
            if tok_type == "Column":
                try:
                    col_id, _ = cls._parse_column(tok_val)
                except:
                    logging.getLogger(cls.__name__).warning(f'unrecognized column {tok_val}')
                    continue
                wsc_graph.add_e_both(sql_node, f"col-{col_id}")

            elif tok_type.startswith("Literal"):
                for word_id in tok_span:
                    assert isinstance(word_id, int)
                    wsc_graph.add_e_both(sql_node, f"word-{word_id}")

    @classmethod
    def _add_w2s_edges(cls, wsc_graph: Graph[str], align):
        for words, sqls in align:
            for word_id in words:
                for sql_id in sqls:
                    wsc_graph.add_e_both(f"word-{word_id}", f"sql-{sql_id}")

    def __init__(self, ns_keyword: str = 'keyword', ns_coltype: str = 'col_type'):
        super().__init__()
        self.ns_keyword: str = ns_keyword
        self.ns_coltype: str = ns_coltype

        self.padded_keys: Dict[int, List[str]] = {
            0: [
                # target keys, target_types are acturally general paddings
                'tgt_type', 'tgt_keyword', 'tgt_col_type',  # prediction-based
                'tgt_col_id', 'tgt_literal_begin', 'tgt_literal_end',   # copy-based
            ],
            -1: [
                # alignment keys
                'align_ws_word', 'align_ws_sql',
                'align_wc_word', 'align_wc_col',
                'align_sc_sql', 'align_sc_col',
            ],
            1: [
                'col_type_mask',
            ]
        }

