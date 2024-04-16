import torch

def focal_loss(output, target, gamma=2, alpha=0.25):
    
    # Add epsilon to avoid overflow when p is 0.0
    epsilon = 0.00001
    p = torch.sigmoid(output) + epsilon
    pt = (1-target) * (1-p) + target * p
    alphat = (1-target) * (1-alpha) + target * alpha
    bce = -torch.log(pt)
    weights = alphat * (1 - pt).pow(gamma)
    loss_arr = weights * bce
    return loss_arr.sum()