from typing import Any, Callable, Dict, List, Optional, Set
import logging
import os

from tensorboardX import SummaryWriter
import torch

# from allennlp.common.from_params import FromParams
# from allennlp.data.dataloader import TensorDict
# from allennlp.nn import util as nn_util
# from allennlp.training.optimizers import Optimizer
# from allennlp.training import util as training_util
# from allennlp.models.model import Model

logger = logging.getLogger(__name__)


class TensorboardWriter:
    """
    Class that handles Tensorboard (and other) logging.

    # Parameters

    serialization_dir : `str`, optional (default = `None`)
        If provided, this is where the Tensorboard logs will be written.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "tensorboard_writer", it gets passed in separately.
    summary_interval : `int`, optional (default = `100`)
        Most statistics will be written out only every this many batches.
    histogram_interval : `int`, optional (default = `None`)
        If provided, activation histograms will be written out every this many batches.
        If None, activation histograms will not be written out.
        When this parameter is specified, the following additional logging is enabled:
            * Histograms of model parameters
            * The ratio of parameter update norm to parameter norm
            * Histogram of layer activations
        We log histograms of the parameters returned by
        `model.get_parameters_for_histogram_tensorboard_logging`.
        The layer activations are logged for any modules in the `Model` that have
        the attribute `should_log_activations` set to `True`.  Logging
        histograms requires a number of GPU-CPU copies during training and is typically
        slow, so we recommend logging histograms relatively infrequently.
        Note: only Modules that return tensors, tuples of tensors or dicts
        with tensors as values currently support activation logging.
    should_log_parameter_statistics : `bool`, optional (default = `True`)
        Whether to log parameter statistics (mean and standard deviation of parameters and
        gradients).
    should_log_learning_rate : `bool`, optional (default = `False`)
        Whether to log (parameter-specific) learning rate.
    get_iteration_fn: `Callable[[], int]`, optional (default = `None`)
        A thunk that returns the number of batches so far. Most likely this will
        be a closure around an instance variable in your `Trainer` class.  Because of circular
        dependencies in constructing this object and the `Trainer`, this is typically `None` when
        you construct the object, but it gets set inside the constructor of our `Trainer`.
    """

    def __init__(
        self,
        serialization_dir: Optional[str] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        get_iteration_fn: Callable[[], int] = None,
    ) -> None:
        if serialization_dir is not None:
            # Create log directories prior to creating SummaryWriter objects
            # in order to avoid race conditions during distributed training.
            train_ser_dir = os.path.join(serialization_dir, "log", "train")
            os.makedirs(train_ser_dir, exist_ok=True)
            self._train_log = SummaryWriter(train_ser_dir)
            val_ser_dir = os.path.join(serialization_dir, "log", "validation")
            os.makedirs(val_ser_dir, exist_ok=True)
            self._validation_log = SummaryWriter(val_ser_dir)
        else:
            self._train_log = self._validation_log = None

        self._summary_interval = summary_interval
        self._histogram_interval = histogram_interval
        self._should_log_parameter_statistics = should_log_parameter_statistics
        self.get_iteration = get_iteration_fn

        self._histogram_parameters: Set[str] = None

    def log_batch(
        self,
        model: torch.nn.Module,
        loss: float,
        metrics: Dict[str, float],
        param_updates: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        if self.should_log_this_batch():
            self.log_parameter_and_gradient_statistics(model)

            self.add_train_scalar("loss/loss_train", loss)
            self.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

        if self.should_log_histograms_this_batch():
            self.log_histograms(model)
            self.log_gradient_updates(model, param_updates)

    def should_log_this_batch(self) -> bool:
        return self.get_iteration() % self._summary_interval == 0

    def should_log_histograms_this_batch(self) -> bool:
        return (
            self._histogram_interval is not None
            and self.get_iteration() % self._histogram_interval == 0
        )

    def add_train_scalar(self, name: str, value: float, timestep: int = None) -> None:
        timestep = timestep or self.get_iteration()
        # get the scalar
        if self._train_log is not None:
            self._train_log.add_scalar(name, _item(value), timestep)

    def add_train_histogram(self, name: str, values: torch.Tensor) -> None:
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write, self.get_iteration())

    def add_validation_scalar(self, name: str, value: float, timestep: int = None) -> None:
        timestep = timestep or self.get_iteration()
        if self._validation_log is not None:
            self._validation_log.add_scalar(name, _item(value), timestep)

    def log_parameter_and_gradient_statistics(self, model: torch.nn.Module) -> None:
        """
        Send the mean and std of all parameters and gradients to tensorboard, as well
        as logging the average gradient norm.
        """
        if self._should_log_parameter_statistics:
            # Log parameter values to Tensorboard
            for name, param in model.named_parameters():
                param_data = param.data.float()
                if param.data.numel() > 0:
                    self.add_train_scalar("parameter_mean/" + name, param_data.mean())
                if param.data.numel() > 1:
                    self.add_train_scalar("parameter_std/" + name, param_data.std())
                if param.grad is not None:
                    if param.grad.is_sparse:

                        grad_data = param.grad.data._values()
                    else:
                        grad_data = param.grad.data

                    grad_data = grad_data.float()

                    # skip empty gradients
                    if torch.prod(torch.tensor(grad_data.shape)).item() > 0:
                        self.add_train_scalar("gradient_mean/" + name, grad_data.mean())
                        if grad_data.numel() > 1:
                            self.add_train_scalar("gradient_std/" + name, grad_data.std())
                    else:
                        # no gradient for a parameter with sparse gradients
                        logger.info("No gradient for %s, skipping tensorboard logging.", name)

    def log_histograms(self, model: torch.nn.Module) -> None:
        """
        Send histograms of parameters to tensorboard.
        """
        for name, param in model.named_parameters():
            self.add_train_histogram("parameter_histogram/" + name, param)

    def log_gradient_updates(self, model: torch.nn.Module, param_updates: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            update_norm = torch.norm(param_updates[name].view(-1).float())
            param_norm = torch.norm(param.view(-1).float()).cpu()
            self.add_train_scalar(
                "gradient_update/" + name,
                update_norm / (param_norm + 1e-31),
            )

    def log_metrics(
        self,
        train_metrics: dict,
        val_metrics: dict = None,
        epoch: int = None,
    ) -> None:
        """
        Sends all of the train metrics (and validation metrics, if provided) to tensorboard.
        """
        metric_names = set(train_metrics.keys())
        if val_metrics is not None:
            metric_names.update(val_metrics.keys())
        val_metrics = val_metrics or {}

        # For logging to the console
        for name in sorted(metric_names):
            # Log to tensorboard
            train_metric = train_metrics.get(name)
            if train_metric is not None:
                self.add_train_scalar(name, train_metric, timestep=epoch)
            val_metric = val_metrics.get(name)
            if val_metric is not None:
                self.add_validation_scalar(name, val_metric, timestep=epoch)

    def enable_activation_logging(self, model: torch.nn.Module) -> None:
        if self._histogram_interval is not None:
            # To log activation histograms to the forward pass, we register
            # a hook on forward to capture the output tensors.
            # This uses a closure to determine whether to log the activations,
            # since we don't want them on every call.
            for _, module in model.named_modules():
                if not getattr(module, "should_log_activations", False):
                    # skip it
                    continue

                def hook(module_, inputs, outputs):

                    log_prefix = "activation_histogram/{0}".format(module_.__class__)
                    if self.should_log_histograms_this_batch():
                        self.log_activation_histogram(outputs, log_prefix)

                module.register_forward_hook(hook)

    def log_activation_histogram(self, outputs, log_prefix: str) -> None:
        if isinstance(outputs, torch.Tensor):
            log_name = log_prefix
            self.add_train_histogram(log_name, outputs)
        elif isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                log_name = "{0}_{1}".format(log_prefix, i)
                self.add_train_histogram(log_name, output)
        elif isinstance(outputs, dict):
            for k, tensor in outputs.items():
                log_name = "{0}_{1}".format(log_prefix, k)
                self.add_train_histogram(log_name, tensor)
        else:
            # skip it
            pass

    def close(self) -> None:
        """
        Calls the `close` method of the `SummaryWriter` s which makes sure that pending
        scalars are flushed to disk and the tensorboard event files are closed properly.
        """
        if self._train_log is not None:
            self._train_log.close()
        if self._validation_log is not None:
            self._validation_log.close()


def _item(value: Any):
    if hasattr(value, "item"):
        val = value.item()
    else:
        val = value
    return val
