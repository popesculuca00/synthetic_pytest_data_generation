def bce_loss(input_, target):
    
    neg_abs = -input_.abs()
    loss = input_.clamp(min=0) - input_ * target + (1 + neg_abs.exp()).log()
    return loss.mean()