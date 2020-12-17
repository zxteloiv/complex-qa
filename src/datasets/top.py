from os.path import join, abspath, dirname
from trialbot.training import TrialBot, Events, Registry
from .field import FieldAwareTranslator
from .seq_field import SeqField

@Registry.dataset('top')
def top_dataset():
    from trialbot.data.datasets.tabular_dataset import TabSepFileDataset
    from .composition_dataset import SequentialToDict
    from trialbot.utils.root_finder import find_root

    def _get_top_data(filename):
        top_path = join(find_root(), 'data', 'top-data', filename)
        return SequentialToDict(TabSepFileDataset(top_path),
                                keys=['utterance', 'tokenized_utterance', 'top_representation'])

    files = ['train.tsv', 'eval.tsv', 'test.tsv']
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
            SeqField(source_key='tokenized_utterance', renamed_key='source_tokens', add_start_end_toks=False),
            SeqField(source_key='top_representation', renamed_key='target_tokens'),
        ])
