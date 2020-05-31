from itertools import cycle, zip_longest

def zip_cycle(*iterables, empty_default=None):
    """same as zip longest, but cycling when short lists are exhausted"""
    cycles = [cycle(i) for i in iterables]
    for _ in zip_longest(*iterables):
        yield tuple(next(i, empty_default) for i in cycles)
