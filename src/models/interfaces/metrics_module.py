from typing import Protocol, runtime_checkable, Dict, Any


@runtime_checkable
class MetricsModule(Protocol):
    """
    If a Module is a subclass of the LossModule, it will generate a loss after forward.
    A general model may consider to check each module it possesses whether it yields a loss.
    """
    def get_metrics(self, reset=False) -> Dict[str, Any]:
        raise NotImplementedError
