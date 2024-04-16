import torch

def estimate_cov(examples, rowvar=False, inplace=False):
    
    if examples.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()