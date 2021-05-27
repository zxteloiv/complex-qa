from trialbot.data import Dataset
import lark
import logging
from typing import List, Union, Tuple, Optional

class LarkGrammarDataset(Dataset):
    def __init__(self, grammar_filename, startpoint, ):
        self.filename = grammar_filename
        self.startpoint = startpoint
        self.data = None
        self.term_defs = dict()

    def read_data(self):
        if self.data is None:
            parser = lark.Lark(open(self.filename), start=self.startpoint, keep_all_tokens=True)
            rules: List[lark.Tree] = parser.rules
            term_defs = parser.terminals

            term_def_lookup = { term_def.name: term_def.pattern for term_def in term_defs }
            self.term_defs.update(term_def_lookup)

            self.data = []
            for r in rules:
                lhs = r.origin.name
                rhs_seq = []
                for t in r.expansion:
                    symbol = {'token': t.name, 'exact_token': None, 'fidelity': 0}
                    if t.is_term:
                        pattern = term_def_lookup[t.name]   # fatal error if t.name is not contained
                        symbol['fidelity'] += 1
                        if pattern.type == 'str':
                            symbol['fidelity'] += 1
                            exact = pattern.value
                            symbol['exact_token'] = exact.lower() if 'i' in pattern.flags else exact
                    rhs_seq.append(symbol)

                self.data.append([lhs, rhs_seq])

    def get_example(self, i: int):
        self.read_data()
        return self.data[i]

    def __len__(self):
        self.read_data()
        return len(self.data)

class LarkParserDatasetWrapper(Dataset):
    PARSER_REG = dict()

    def __init__(self, grammar_filename, startpoint, parse_keys: Union[str, List[str], None], dataset: Dataset):
        super().__init__()
        self.parser = self.init_parser(grammar_filename, startpoint)
        self.grammar_file = grammar_filename
        self.startpoint = startpoint
        self.dataset = dataset
        self.keys = parse_keys
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def init_parser(cls, grammar_filename, startpoint):
        if grammar_filename is None:
            logging.warning(f"grammar file not given, lark dataset wrapper used as passthrough")
            return None

        if grammar_filename in cls.PARSER_REG:
            return cls.PARSER_REG[grammar_filename]
        else:
            parser = lark.Lark(open(grammar_filename), start=startpoint, keep_all_tokens=True)
            cls.PARSER_REG[grammar_filename] = parser
            return parser

    def __len__(self):
        return len(self.dataset)

    def __getstate__(self):
        state = self.__dict__.copy()
        # the XEarley parser will not be able to serialize, so we have to initialize it runtime at the subprocess
        del state['parser']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.parser = self.init_parser(self.grammar_file, self.startpoint)

    def get_example(self, i: int):
        example: dict = self.dataset.get_example(i)
        tree_dict = self._parse(example)
        example.update(tree_dict)
        return example

    def _parse(self, example):
        parse_tree = {}
        if self.parser is None:
            return parse_tree

        for key in self.keys:
            try:
                t = self.parser.parse(example[key])
            except KeyboardInterrupt:
                raise SystemExit("Received Keyboard Interupt and Exit now.")
            except:
                self.logger.warning(f'Failed to parse the key {key} for the example {str(example)}, set as None')
                t = None
            parse_tree[key + '_tree'] = t
        return parse_tree

