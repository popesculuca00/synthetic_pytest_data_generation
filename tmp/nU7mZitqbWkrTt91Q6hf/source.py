def average_loss(losses, mask=None):
    

    if mask is not None:
        assert mask.size() == losses.size(), 'mask must be the same size as losses'
        losses = losses * mask
        denom = mask.sum()
    else:
        denom = losses.numel()

    # Prevent division by zero
    if isinstance(denom, int):
        denom = max(denom, 1)
    else:
        denom = denom.clamp(1)

    return losses.sum() / denom