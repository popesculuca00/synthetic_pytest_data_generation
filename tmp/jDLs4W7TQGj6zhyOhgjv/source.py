import torch

def gradient_to_excitation_backprop_saliency(x):
    r
    return torch.sum(x.grad, 1, keepdim=True)