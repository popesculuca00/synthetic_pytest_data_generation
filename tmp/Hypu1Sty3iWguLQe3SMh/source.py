import torch

def hat_inv(h: torch.Tensor):
    

    N , dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError('Input has to be a batch of 3x3 Tensors.')

    ss_diff = (h + h.permute(0, 2, 1)).abs().max()
    if float(ss_diff) > 1e-5:
        raise ValueError('One of input matrices not skew-symmetric.')

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v