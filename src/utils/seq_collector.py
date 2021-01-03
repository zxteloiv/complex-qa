from typing import Optional
from collections import defaultdict
import logging

class SeqCollector:
    def __init__(self, history: Optional[defaultdict] = None):
        self.history: defaultdict[str, list] = history or defaultdict(list)

    def __call__(self, *args, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def __iter__(self):
        return iter(self.history)

    def __getitem__(self, item):
        return self.history[item]

    def get_stacked_tensor(self, tag, default=None, dim=1):
        import torch
        seq_data = self.history[tag]
        if len(seq_data) == 0:
            return default

        if isinstance(seq_data[0], torch.Tensor):
            return torch.stack(seq_data, dim=dim)
        else:
            return seq_data

    def get_concatenated_tensor(self, tag, default=None, dim=-1):
        import torch
        seq_data = self.history[tag]
        if len(seq_data) == 0:
            return default

        if isinstance(seq_data[0], torch.Tensor):
            return torch.cat(seq_data, dim=dim)
        else:
            return seq_data

    def remember_func(self, input_tag=None, output_tag=None):
        def deco(func):
            def wrapper(func_obj, *args, **kwargs):
                if input_tag is not None:
                    self(**{input_tag: (args, kwargs)})

                out = func(func_obj, *args, **kwargs)

                if output_tag is not None:
                    self(**{output_tag: out})

                return out
            return wrapper
        return deco

    def validate(self) -> bool:
        pass

if __name__ == '__main__':
    tracker = SeqCollector()
    class A:
        def __init__(self, x=0):
            self.x = x

        @tracker.remember_func(input_tag='A.pow.in', output_tag="A.pow.out")
        def pow(self, p):
            return self.x ** p

        @tracker.remember_func(output_tag='A.add.out')
        def add(self, a):
            return self.x + a

        def __str__(self):
            return str(self.x)

    a = A(3)
    print(a, "** 3 =", a.pow(3))
    print(a, "** 2 =", a.pow(2))
    print(a, "+ 11 =", a.add(11))

    print("in history:")
    for k in tracker:
        print(k, tracker[k])




