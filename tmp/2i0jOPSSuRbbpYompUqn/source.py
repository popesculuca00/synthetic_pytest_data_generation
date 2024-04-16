import torch

def forward_fill(x, fill_index=-2):
    
    # Checks
    assert isinstance(x, torch.Tensor)
    assert x.dim() >= 2

    mask = torch.isnan(x)
    if mask.any():
        cumsum_mask = (~mask).cumsum(dim=fill_index)
        cumsum_mask[mask] = 0
        _, index = cumsum_mask.cummax(dim=fill_index)
        x = x.gather(dim=fill_index, index=index)

    return x