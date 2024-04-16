import torch

def normalize(tensor, mean, std, inplace=False):
    
    if not torch.is_tensor(tensor):
        raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndimension() != 3:
        raise ValueError('Expected tensor to be a tensor image of size (C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor