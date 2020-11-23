from trialbot.data import Dataset
import lark
from typing import List

class LarkDataset(Dataset):
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
                    symbol = {'token': t.name, 'exact_token': None, 'exactitude': 0}
                    if t.is_term:
                        pattern = term_def_lookup[t.name]   # fatal error if t.name is not contained
                        symbol['exactitude'] += 1
                        if pattern.type == 'str':
                            symbol['exactitude'] += 1
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
