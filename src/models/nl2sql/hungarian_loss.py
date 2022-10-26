from scipy.optimize import linear_sum_assignment
import torch


def get_hungarian_sup_loss(attn_weights, vec_mask, mat_mask):
    bsz, vec_len, mat_len = attn_weights.size()
    hungarian_target = torch.zeros_like(attn_weights)
    for b in range(bsz):
        attn_matrix = attn_weights[b].detach().cpu().numpy()
        rows, cols = linear_sum_assignment(attn_matrix, maximize=True)
        hungarian_target[b][rows, cols] = 1

    pad_mask = vec_mask.unsqueeze(-1) * mat_mask.unsqueeze(-2)  # (b, #vec, #mat)
    l2_loss = (attn_weights - hungarian_target) ** 2 / 2 * pad_mask
    return l2_loss.sum(dim=(1, 2)).mean()


def get_hungarian_reg_loss(attn_weights, vec_mask, mat_mask):
    bsz, vec_len, mat_len = attn_weights.size()
    dev = attn_weights.device
    hungarian_target = torch.zeros_like(attn_weights)
    for b in range(bsz):
        attn_matrix = attn_weights[b].detach().cpu().numpy()
        rows, cols = linear_sum_assignment(attn_matrix, maximize=True)
        hungarian_target[b][rows, cols] = 1

    max_pos = attn_weights.argmax(dim=-1)  # (bsz, #vec)
    max_onehot = torch.zeros_like(attn_weights)  # (bsz, #vec, #mat)

    def tr(*args, **kwargs): return torch.arange(*args, **kwargs, device=dev)

    max_onehot[tr(bsz).view(-1, 1), tr(vec_len).view(1, -1), max_pos] = 1
    pad_mask = vec_mask.unsqueeze(-1) * mat_mask.unsqueeze(-2)  # (b, #vec, #mat)

    # if a hungarian target is already the same as softmax onehot, no need to train it anymore
    reg_mask = (hungarian_target != max_onehot)

    raw_l2_loss = (attn_weights - hungarian_target) ** 2 / 2 * pad_mask * reg_mask
    return raw_l2_loss.sum(dim=(1, 2)).mean()
