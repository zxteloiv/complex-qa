from torch import nn


class DummyModel(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    def forward(self, *args, **kwargs):
        return self.obj(*args, **kwargs)

    @classmethod
    def factory_of_init(cls, obj):
        def init_func(*args, **kwargs):
            return cls(obj)     # ignore every argument
        return init_func

