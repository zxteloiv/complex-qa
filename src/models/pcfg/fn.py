from torch.utils.checkpoint import checkpoint as ckp

def checkpoint(func):
    """ To save memory of the inside algorithm. """
    def wrapper(*args, **kwargs):
        return ckp(func, *args, **kwargs)
    return wrapper


def stripe(x, n, w, offset=(0, 0), dim=1):
    """
    I borrow the idea from:
    https://github.com/yzhangcs/parser/blob/a8e6f443febf8d986cd6eba74966fdf924cb567d/supar/utils/fn.py#L32
    to implement `stripe' function. If you do not understand the code, plz refer to their code and comment.
    Roughly speaking, this function packs all inside scores of spans of same width w, which facilitates parallel computation.
    """
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    else:
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def diagonal_copy_(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(size=(x.shape[0], seq_len - w,  *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(y)
    else:
        x.as_strided(size=(x.shape[0], seq_len - w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(y)


def diagonal(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )
    else:
        return x.as_strided(size=(x.shape[0], seq_len - w),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )

