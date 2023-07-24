from functools import partial


def select_optim(p):
    from trialbot.training.select_optims import torch_optim_cls
    try:
        cls = torch_optim_cls(p)
        return cls
    except ValueError:
        pass

    cls = user_optim_cls(p)
    return cls


def user_optim_cls(p):
    name = getattr(p, "OPTIM", "adabelief").lower()
    kwargs = p.OPTIM_KWARGS

    if name == "adsgd":
        from optim.adsgd import AdSGD
        cls = partial(AdSGD, lr=p.SGD_LR, weight_decay=p.WEIGHT_DECAY, **kwargs)

    elif name == "eadam":
        from optim.EAdam import EAdam
        cls = partial(EAdam, lr=p.ADAM_LR, betas=p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY, **kwargs)

    elif name == 'adabelief':
        from adabelief_pytorch import AdaBelief
        cls = partial(AdaBelief, lr=p.ADAM_LR, betas=p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY,
                      print_change_log=False, **kwargs)

    else:
        raise ValueError(f"Optim {name} not found.")

    return cls

