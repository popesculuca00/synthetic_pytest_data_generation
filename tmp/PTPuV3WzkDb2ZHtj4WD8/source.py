def bce_loss(input, target, reduce=True):
    
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    if reduce:
        return loss.mean()
    else:
        return loss