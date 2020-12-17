from os.path import join, abspath, dirname
from trialbot.training import TrialBot, Events, Registry
from .field import FieldAwareTranslator
from .seq_field import SeqField


def _get_top_data(filename):
    from trialbot.data.datasets.tabular_dataset import TabSepFileDataset
    from .composition_dataset import SequentialToDict
    from trialbot.utils.root_finder import find_root

    top_path = join(find_root(), 'data', 'top-data', filename)
    return SequentialToDict(TabSepFileDataset(top_path),
                            keys=['utterance', 'tokenized_utterance', 'top_representation'])

@Registry.dataset('top')
def top_dataset():
    files = ['train.tsv', 'eval.tsv', 'test.tsv']
    return tuple(map(_get_top_data, files))

@Registry.dataset('top_filtered')
def top_dataset():
    files = ['train_filtered.tsv', 'eval_filtered.tsv', 'test_filtered.tsv']
    return tuple(map(_get_top_data, files))

@Registry.translator('top')
class TopTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='tokenized_utterance', renamed_key='source_tokens', add_start_end_toks=False),
            SeqField(source_key='top_representation', renamed_key='target_tokens'),
        ])

@Registry.translator('top_unified')
class TopUnifiedVocabTranslator(FieldAwareTranslator):
    def __init__(self):
        super().__init__(field_list=[
            SeqField(source_key='tokenized_utterance', renamed_key='source_tokens', namespace="unified_vocab", add_start_end_toks=False),
            SeqField(source_key='top_representation', renamed_key='target_tokens', namespace="unified_vocab"),
        ])
