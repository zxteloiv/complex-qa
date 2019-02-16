from typing import List, Tuple, Dict, Mapping, Optional
import torch


def add_position_and_timestep_sinusoid(inputs: torch.Tensor,
                                       timestep: Optional[float] = None,
                                       base_num: float = 1.e4) -> torch.Tensor:
    """
    Add positional embedding to inputs, which contains words at every position

    :param inputs: input embedding for entire sequence, a shape of (batch, seq_len, embedding) assumed
    :param timestep: which timestep is the current recurrent block

    :param base_num: the base number used in dimensional power, default to 10^4 as the Attention is All You Need paper.

    :return: same shape as inputs
    """
    _, seq_len, emb_dim = inputs.size()
    assert emb_dim % 2 == 0, "embedding dimension must be even"

    # position: [0, 1, ..., seq_len - 1] with shape (seq_len, )
    # negative_half_dim_index: [-0, -1, ..., -(emb_dim // 2 - 1)] with shape (emb_dim // 2, )
    # inverse_dim: [base_num ^(-0), ..., base_num^(-(emb_dim // 2 - 1))], with shape (emb_dim // 2,)
    position: torch.Tensor = inputs.new_ones(seq_len).cumsum(dim=0).float() - 1
    negative_half_dim_index: torch.Tensor = -(inputs.new_ones(emb_dim // 2).cumsum(dim=0).float() - 1)
    inverse_dim: torch.Tensor = torch.pow(base_num, negative_half_dim_index)

    # x: (seq_len, emb_dim // 2) <- (seq_len, 1) * (1, emb_dim // 2)
    x: torch.Tensor = position.unsqueeze(1) * inverse_dim.unsqueeze(0)

    if timestep is not None:
        # y: (1, emb_dim // 2)
        y: torch.Tensor = inverse_dim.unsqueeze(0) * timestep
        sinusoid_odd, sinusoid_even = x.sin() + y.sin(), x.cos() + y.cos()
    else:
        sinusoid_odd, sinusoid_even = x.sin(), x.cos()

    # sinusoid: (seq_len, emb_dim // 2, 2) -> (1, seq_len, emb_dim)
    sinusoid: torch.Tensor = torch.stack([sinusoid_odd, sinusoid_even], dim=2).reshape(1, seq_len, -1)

    return inputs + sinusoid

def add_positional_features(inputs: torch.Tensor) -> torch.Tensor:
    """
    A wrapper with the same name from AllenNLP
    """
    return add_position_and_timestep_sinusoid(inputs, None)
