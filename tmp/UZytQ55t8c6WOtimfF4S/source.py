def dice_coef(inputs, targets, smooth=1.0):
    
    inputs = inputs.sigmoid().flatten(2).unsqueeze(2)
    targets = targets.flatten(2).unsqueeze(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    coef = (numerator + smooth) / (denominator + smooth)
    coef = coef.mean(0)  # average on the temporal dim to get instance trajectory scores
    return coef