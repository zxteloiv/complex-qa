from itertools import cycle, zip_longest

def zip_cycle(*iterables, empty_default=None):
    """same as zip longest, but cycling when short lists are exhausted"""
    cycles = [cycle(i) for i in iterables]
    for _ in zip_longest(*iterables):
        yield tuple(next(i, empty_default) for i in cycles)

from itertools import chain, combinations

def powerset(iterable, min_len: int = 0, max_len: int = None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(min_len, (len(s)+1) if max_len is None else max_len))