from typing import Callable
from .hparamset import HyperParamSet
from .runner import ExperimentRunner
from allennlp.models import Model
from allennlp.data import Vocabulary

def run(name: str, get_model_func: Callable[[HyperParamSet, Vocabulary], Model]):
    runner = ExperimentRunner(name, get_model_func=get_model_func)
    runner.run()
