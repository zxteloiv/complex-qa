from typing import Callable, Optional, Any, Union
import math
import multiprocessing
import random
import sys
import time
from functools import partial

KeyFuncType = Callable[[Any], Union[float, int]]

def merge(*args, key: Optional[KeyFuncType]):
    keyfn = (lambda x: x) if key is None else key
    # Support explicit left/right args, as well as a two-item
    # tuple which works more cleanly with multiprocessing.
    left, right = args[0] if len(args) == 1 else args
    left_length, right_length = len(left), len(right)
    left_index, right_index = 0, 0
    merged = []
    while left_index < left_length and right_index < right_length:
        if keyfn(left[left_index]) <= keyfn(right[right_index]):
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    if left_index == left_length:
        merged.extend(right[right_index:])
    else:
        merged.extend(left[left_index:])
    return merged


def merge_sort(data, key: Optional[KeyFuncType] = None):
    length = len(data)
    if length <= 1:
        return data
    middle = length // 2
    left = merge_sort(data[:middle], key)
    right = merge_sort(data[middle:], key)
    return merge(left, right, key=key)


def merge_sort_parallel(data, key: Optional[KeyFuncType] = None, max_pool_size: int = 8, pool = None):
    # Creates a pool of worker processes, one per CPU core.
    # We then split the initial data into partitions, sized
    # equally per worker, and perform a regular merge sort
    # across each partition.
    processes = min(multiprocessing.cpu_count(), max_pool_size)
    pool = pool or multiprocessing.Pool(processes=processes)
    size = int(math.ceil(float(len(data)) / processes))
    data = [data[i * size:(i + 1) * size] for i in range(processes)]
    keyed_merge_sort = partial(merge_sort, key=key)
    keyed_merge = partial(merge, key=key)
    data = pool.map(keyed_merge_sort, data)
    # Each partition is now sorted - we now just merge pairs of these
    # together using the worker pool, until the partitions are reduced
    # down to a single sorted result.
    while len(data) > 1:
        # If the number of partitions remaining is odd, we pop off the
        # last one and append it back after one iteration of this loop,
        # since we're only interested in pairs of partitions to merge.
        extra = data.pop() if len(data) % 2 == 1 else None
        data = [(data[i], data[i + 1]) for i in range(0, len(data), 2)]
        data = pool.map(keyed_merge, data) + ([extra] if extra else [])
    return data[0]


if __name__ == "__main__":
    size = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 1000
    data_unsorted = [random.randint(0, size) for _ in range(size)]
    for sort in merge_sort, merge_sort_parallel:
        start = time.time()
        def keyfunc(x):
            return x ** 2
        data_sorted = sort(data_unsorted, key=keyfunc)
        end = time.time() - start
        print(sort.__name__, end, sorted(data_unsorted) == data_sorted)