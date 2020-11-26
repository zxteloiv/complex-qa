from typing import Optional
from trialbot.training import Registry
from trialbot.data.datasets.jsonl_dataset import JsonLDataset
from .lark_dataset import LarkGrammarDataset, LarkParserDatasetWrapper
from trialbot.data.datasets.index_dataset import IndexDataset
from utils.root_finder import find_root
import json
from os.path import join

ROOT = find_root()
CFQ_PATH = join(ROOT, 'data', 'cfq')
GRAMMAR_PATH = join(ROOT, 'src', 'statics', 'grammar')

def cfq_filebase(split_filename):
    store = JsonLDataset(join(CFQ_PATH, 'dataset_slim.jsonl'))
    all_idx = json.load(open(split_filename))
    split_idx = list(map(all_idx.get, ['trainIdxs', 'devIdxs', 'testIdxs']))
    return tuple(IndexDataset(store, ds) for ds in split_idx)

def cfq_treebase(split_filename, grammar_file: str, keys):
    store = JsonLDataset(join(CFQ_PATH, 'dataset_slim.jsonl'))
    all_idx = json.load(open(split_filename))
    split_idx = list(map(all_idx.get, ['trainIdxs', 'devIdxs', 'testIdxs']))
    get_parse_tree_dataset = lambda d: LarkParserDatasetWrapper(grammar_file, 'queryunit', keys, d)
    return tuple(get_parse_tree_dataset(IndexDataset(store, ds)) for ds in split_idx)

class PatchedJsonLDataset(JsonLDataset):
    def _read_data(self):
        import trialbot.utils.file_reader as reader_utils
        if self._data is None:
            self._data = list(line.rstrip('\r\n') if isinstance(line, str) else line.decode().rstrip('\r\n') for line in reader_utils.open_file(self.filename))

def complete_cfq():
    return PatchedJsonLDataset(join(CFQ_PATH, 'dataset_slim.jsonl.gz'))

@Registry.dataset()
def cfq_mcd1():
    splitfile = join(CFQ_PATH, 'splits', 'mcd1.json')
    return cfq_filebase(splitfile)

@Registry.dataset()
def cfq_mcd1_tree():
    splitfile = join(CFQ_PATH, 'splits', 'mcd1.json')
    grammar = join(GRAMMAR_PATH, 'sparql_pattern.bnf.lark')
    return cfq_treebase(splitfile, grammar, ['sparql', 'sparqlPattern', 'sparqlPatternModEntities'])

@Registry.dataset('sparql')
def sparql_grammar(filename='sparql.bnf.lark'):
    file = join(GRAMMAR_PATH, filename)
    d = LarkGrammarDataset(file, 'queryunit')
    return d, d, d

@Registry.dataset('sparql_pattern')
def sparql_pattern_grammar():
    return sparql_grammar('sparql_pattern.bnf.lark')

