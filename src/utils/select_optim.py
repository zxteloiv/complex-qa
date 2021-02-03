import torch

def select_optim(p, params):
    name = getattr(p, "OPTIM", "adamw").lower()
    if name == "sgd":
        optim = torch.optim.SGD(params, p.SGD_LR,
                                weight_decay=p.WEIGHT_DECAY,
                                momentum=getattr(p, 'momentum', 0.),
                                nesterov=getattr(p, 'use_nesterov_sgd', False),
                                )
    elif name == "radam":
        from radam import RAdam
        optim = RAdam(params, lr=p.ADAM_LR, weight_decay=p.WEIGHT_DECAY)
    elif name == "adamw":
        optim = torch.optim.AdamW(params, p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY)
    elif name == "rmsprop":
        optim = torch.optim.RMSprop(params, p.SGD_LR, weight_decay=p.WEIGHT_DECAY)
    elif name == "lbfgs":
        optim = torch.optim.LBFGS(params, getattr(p, 'lr', 1))
    elif name == "eadam":
        from optim.EAdam import EAdam
        optim = EAdam(params, lr=p.ADAM_LR, betas=p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY)
    else:
        optim = torch.optim.Adam(params, p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY)
    return optim

