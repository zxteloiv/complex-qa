from torch import nn
from torch.nn import Sequential

class MultiInputsSequential(Sequential):
    def forward(self, *args):
        for i, module in enumerate(self):
            args = module(*args) if i == 0 else module(args)

        return args


class UnpackedInputsSequential(Sequential):
    def forward(self, *args):
        for module in self:
            args = module(*args)
            if not isinstance(args, tuple):
                args = (args,)

        return args


class SelectArgsById(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def forward(self, *args):
        return args[self.i]


