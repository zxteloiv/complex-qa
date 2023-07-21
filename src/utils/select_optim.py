import torch


def select_optim(p, params):
    name = getattr(p, "OPTIM", "adamw").lower()
    if name == "sgd":
        optim = torch.optim.SGD(params, p.SGD_LR, weight_decay=p.WEIGHT_DECAY,
                                **getattr(p, 'optim_kwargs', dict()),
                                )
    elif name == "adsgd":
        from optim.adsgd import AdSGD
        optim = AdSGD(params, lr=p.SGD_LR, weight_decay=p.WEIGHT_DECAY,
                      **getattr(p, 'optim_kwargs', dict()), )
    elif name == "radam":
        optim = torch.optim.RAdam(params, lr=p.ADAM_LR, betas=p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY,
                                  **getattr(p, 'optim_kwargs', dict()),)
    elif name == "adamw":
        optim = torch.optim.AdamW(params, p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY,
                                  **getattr(p, 'optim_kwargs', dict()),)
    elif name == "rmsprop":
        optim = torch.optim.RMSprop(params, p.SGD_LR, weight_decay=p.WEIGHT_DECAY,
                                    **getattr(p, 'optim_kwargs', dict()))
    elif name == "lbfgs":
        optim = torch.optim.LBFGS(params, getattr(p, 'lr', 1), **getattr(p, 'optim_kwargs', dict()))
    elif name == "eadam":
        from optim.EAdam import EAdam
        optim = EAdam(params, lr=p.ADAM_LR, betas=p.ADAM_BETAS,
                      weight_decay=p.WEIGHT_DECAY,
                      **getattr(p, 'optim_kwargs', dict()))
    elif name == 'adabelief':
        from adabelief_pytorch import AdaBelief
        optim = AdaBelief(params, p.ADAM_LR, p.ADAM_BETAS,
                          weight_decay=p.WEIGHT_DECAY,
                          print_change_log=False,
                          **getattr(p, 'optim_kwargs', dict()))
    elif name == "adam":
        optim = torch.optim.Adam(params, p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY,
                                 **getattr(p, 'optim_kwargs', dict()))
    elif name == "adamax":
        optim = torch.optim.Adamax(params, p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY,
                                   **getattr(p, 'optim_kwargs', dict()))
    else:
        raise ValueError(f"Optim {name} not found.")
    return optim

