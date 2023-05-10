from trialbot.data import Dataset
import lark
import logging
from typing import List, Union, Tuple, Optional


class LarkParserDatasetWrapper(Dataset):
    PARSER_REG = dict()

    def __init__(self, grammar_filename, startpoint, parse_keys: Union[str, List[str], None], dataset: Dataset):
        super().__init__()
        self.parser = self.init_parser(grammar_filename, startpoint)
        self.grammar_file = grammar_filename
        self.startpoint = startpoint
        self.dataset = dataset
        self.keys = [parse_keys] if isinstance(parse_keys, str) else parse_keys
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

