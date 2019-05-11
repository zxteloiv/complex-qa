from typing import Callable
from config import HyperParamSet
from .runner import ExperimentRunner
from allennlp.models import Model
from allennlp.data import Vocabulary

def run(name: str, get_model_func: Callable[[HyperParamSet, Vocabulary], Model]):
    class MyExperiment(ExperimentRunner):
        def get_model(self, hparams, vocab):
            return get_model_func(hparams, vocab)
    runner = MyExperiment(name)
    runner.run()
