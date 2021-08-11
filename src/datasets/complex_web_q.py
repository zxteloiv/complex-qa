from trialbot.training import Registry
from trialbot.data.datasets.json_dataset import JsonDataset
from trialbot.utils.root_finder import find_root

import os.path
_ds_path = os.path.join(find_root(), 'data', 'ComplexWebQ')

@Registry.dataset('CompWebQ')
def complex_web_q():
    train_data = JsonDataset(os.path.join(_ds_path, 'ComplexWebQuestions_train.json'))
    dev_data = JsonDataset(os.path.join(_ds_path, 'ComplexWebQuestions_dev.json'))
    test_data = JsonDataset(os.path.join(_ds_path, 'ComplexWebQuestions_test.json'))
    return train_data, dev_data, test_data



