import torch

def select_optim(p, params):
    name = getattr(p, "OPTIM", "adamw").lower()
    if name == "sgd":
        optim = torch.optim.SGD(params, p.SGD_LR, weight_decay=p.WEIGHT_DECAY,
                                **getattr(p, 'optim_kwargs', dict()),
                                )
    elif name == "radam":
        from radam import RAdam
        optim = RAdam(params, lr=p.ADAM_LR, betas=p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY,
                      **getattr(p, 'optim_kwargs', dict()),
                      )
    elif name == "adamw":
        optim = torch.optim.AdamW(params, p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY,
                                  **getattr(p, 'optim_kwargs', dict()),
                                  )
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
                          **getattr(p, 'optim_kwargs', dict()))
    elif name == 'mod_adabelief':
        from optim.ModAdaBelief import AdaBelief
        optim = AdaBelief(params, p.ADAM_LR, p.ADAM_BETAS,
                          weight_decay=p.WEIGHT_DECAY,
                          **getattr(p, 'optim_kwargs', dict()))
    elif name == "ranger_adabelief":
        from ranger_adabelief import RangerAdaBelief
        optim = RangerAdaBelief(params, p.ADAM_LR,
                                betas=p.ADAM_BETAS,
                                weight_decay=p.WEIGHT_DECAY,
                                **getattr(p, 'optim_kwargs', dict()))
    else:
        optim = torch.optim.Adam(params, p.ADAM_LR, p.ADAM_BETAS, weight_decay=p.WEIGHT_DECAY)
    return optim

