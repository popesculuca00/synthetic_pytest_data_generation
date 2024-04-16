import torch

def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    

    if seed is None:
        torch.random.seed()
    else:
        torch.random.manual_seed(seed)
    w = torch.randn(size=shape, dtype=dtype)
    out = w.normal_(mean=mean, std=stddev)
    return out