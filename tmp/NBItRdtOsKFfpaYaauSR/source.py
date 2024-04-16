import torch

def get_backward_gradient(pred_y, y):
    r

    assert isinstance(pred_y, torch.Tensor)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long, device=pred_y.device)
    assert isinstance(y, torch.Tensor)

    if y.shape == pred_y.shape:
        return y
    assert y.dtype == torch.long

    nspatial = len(pred_y.shape) - 2
    grad = torch.zeros_like(pred_y)
    y = y.reshape(-1, 1, *((1,) * nspatial)).expand_as(grad)
    grad.scatter_(1, y, 1.)
    return grad