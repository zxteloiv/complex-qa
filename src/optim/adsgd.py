import numpy as np
import math
import torch
from torch.optim.optimizer import Optimizer

class AdSGD(Optimizer):
    """
    Reimplementation of Adaptive Gradient Descent without Descent.
    Based on https://arxiv.org/abs/1910.09529
    """
    def __init__(self, params, lr=1e-3, eps=1e-16, amplifier=.02, weight_decay=0, damping=1.):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, damping=damping, amplifier=amplifier)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = 0
                state['prev_lr'] = None
                state['prev_param'] = torch.zeros_like(p.data)
                state['prev_grad'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            damping = group['damping']
            amplifier = group['amplifier']

            for p in group['params']:
                if p.grad is None:
                    continue

                data = p.data
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev_lr'] = None
                    state['prev_param'] = torch.zeros_like(data)
                    state['prev_grad'] = torch.zeros_like(data)
                    state['prev_theta'] = torch.zeros_like(data)

                step = state['step']
                if step == 0:
                    state['prev_lr'] = lr
                    state['prev_param'] = data
                    state['prev_grad'] = grad

                    if weight_decay != 0:
                        grad.add_(data, alpha=weight_decay)
                    data.add_(grad, alpha=-lr)

                else:
                    p_norm = (data - state['prev_param']).norm().item() ** 2
                    g_norm = (grad - state['prev_grad']).norm().item() ** 2
                    denom = damping * g_norm
                    norm_div = p_norm / (denom if denom > 0 else eps)

                    if step == 1:
                        new_lr = norm_div + eps
                    else:
                        lr_step = math.sqrt(1 + amplifier * state['prev_theta']) * state['prev_lr']
                        new_lr = min(lr_step, norm_div) + eps

                    state['prev_param'] = data
                    state['prev_grad'] = grad
                    state['prev_lr'] = new_lr
                    state['prev_theta'] = new_lr / state['prev_lr']

                    if weight_decay != 0:
                        grad.add_(data, alpha=weight_decay)
                    data.add_(grad, alpha=-new_lr)

                state['step'] += 1
        return loss
