from typing import List, Mapping, Callable
from trialbot.training import Registry
from itertools import product


def expand_hparams(grid_conf: Mapping[str, list]):
    keys, value_lists = zip(*grid_conf.items())
    for v in enumerate(product(*value_lists)):
        params = dict(zip(keys, v))
        yield params

def update_registry_params(name_prefix, i, params, base_names_fn: Callable):
    name = f"{name_prefix}{i}"

    def hparams_wrapper_fn():
        p = base_names_fn()
        for k, v in params.items():
            setattr(p, k, v)
        return p

    Registry._hparamsets[name] = hparams_wrapper_fn
    return name

def import_grid_search_parameters(grid_conf: Mapping[str, list], base_names_fn: Callable, name_prefix: str = None):
    param_sets = list(expand_hparams(grid_conf))
    import re
    if name_prefix is None:
        name_prefix = re.sub('[^a-zA-Z_0-9]', '', base_names_fn.__name__) + '_'
    names = [update_registry_params(name_prefix, i, params, base_names_fn) for i, params in enumerate(param_sets)]
    return names

if __name__ == '__main__':
    grid_conf = {"batch_size": [64, 128, 256], "hidden_size": [300, 600]}
    from trialbot.training.hparamset import HyperParamSet
    names = import_grid_search_parameters(grid_conf, lambda: HyperParamSet())
    print(f"names={names}")
    print(Registry._hparamsets)
    p = Registry.get_hparamset(names[-1])
    print(type(p))

    for name in names:
        print('--------' * 10)
        print("hparamset: " + name)
        print(Registry.get_hparamset(name))

