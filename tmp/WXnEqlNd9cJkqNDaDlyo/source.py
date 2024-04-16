def featurewise_norm(x, mean=None, std=None, epsilon=1e-7):
    
    if mean:
        x = x - mean
    if std:
        x = x / (std + epsilon)
    return x