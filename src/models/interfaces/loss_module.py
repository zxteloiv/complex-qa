from typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class LossModule(Protocol):
    """
    If a Module is a subclass of the LossModule, it will generate a loss after forward.
    A general model may consider to check each module it possesses whether it yields a loss.
    """
    def get_loss(self) -> None | torch.Tensor:
        raise NotImplementedError
