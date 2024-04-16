import torch

def smooth_l1_loss(pred, target, beta=1.0):
    
    if target.numel() == 0:
        return pred.sum() * 0

    assert beta > 0
    assert pred.size() == target.size()

    diff = (pred - target).abs()
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)

    return loss