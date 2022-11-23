from scipy.optimize import linear_sum_assignment
import torch
from utils.nn import masked_reducing_gather


def get_hungarian_target_by_weight(attn_weights, maximize=True):
    # attn_weights: (b, #vec, #mat)
    bsz, vec_len, mat_len = attn_weights.size()
    hungarian_target = torch.zeros_like(attn_weights)
    for b in range(bsz):
        attn_matrix = attn_weights[b].detach().cpu().numpy()
        rows, cols = linear_sum_assignment(attn_matrix, maximize=maximize)
        hungarian_target[b][rows, cols] = 1
    return hungarian_target     # (b, #vec, #mat)


def get_weight_argmax_onehot(attn_weights):
    bsz, vec_len, mat_len = attn_weights.size()
    max_pos = attn_weights.argmax(dim=-1)  # (bsz, #vec)
    max_onehot = torch.zeros_like(attn_weights, dtype=torch.long)  # (bsz, #vec, #mat)

    def tr(*args, **kwargs):
        return torch.arange(*args, **kwargs, device=attn_weights.device)

    max_onehot[tr(bsz).view(-1, 1), tr(vec_len).view(1, -1), max_pos] = 1
    return max_onehot


def reverse_probabilities(attn_weights, vec_mask, mat_mask):
    # reverse and re-normalize the given probabilistic weights
    n = mat_mask.sum(-1, keepdims=True).unsqueeze(-2)   # (b, 1, 1)
    rev_weights = (1 - attn_weights) / (n - 1).clamp(min=1) * mat_mask.unsqueeze(-2)
    return rev_weights


def hungarian_xent_loss(attn_weights, target, vec_mask, mat_mask, *, regularized: bool = False):
    if regularized:
        max_onehot = get_weight_argmax_onehot(attn_weights)
        # if a hungarian target is already the same as softmax onehot, no need to train it anymore
        reg_mask = (target != max_onehot)  # (b, #vec, #mat)
        mask = (reg_mask.sum(-1) > 0) * vec_mask  # (b, #vec)

    else:
        mask = vec_mask

    xent_loss = - masked_reducing_gather(attn_weights.clamp(min=1e-8).log(), target.argmax(dim=-1), mask, 'batch')
    return xent_loss


def hungarian_l2_loss(attn_weights, target, vec_mask, mat_mask, *, regularized: bool = False):
    pad_mask = vec_mask.unsqueeze(-1) * mat_mask.unsqueeze(-2)  # (b, #vec, #mat)
    l2_loss = (attn_weights - target) ** 2 / 2 * pad_mask

    if regularized:
        max_onehot = get_weight_argmax_onehot(attn_weights)
        reg_mask = (target != max_onehot)
        l2_loss = l2_loss * reg_mask
    return l2_loss.sum(dim=(1, 2)).mean()


# ---------------------
# wrapper functions
# ---------------------

def get_hungarian_sup_loss(attn_weights, vec_mask, mat_mask):
    hungarian_target = get_hungarian_target_by_weight(attn_weights)
    return hungarian_l2_loss(attn_weights, hungarian_target, vec_mask, mat_mask)


def get_hungarian_reg_loss(attn_weights, vec_mask, mat_mask):
    hungarian_target = get_hungarian_target_by_weight(attn_weights)
    return hungarian_l2_loss(attn_weights, hungarian_target, vec_mask, mat_mask, regularized=True)
