def batch_to_single_instance(X):
    
    x = X[0]
    assert False, x.ndim
    if x.ndim == 0 and x.dtype == 'int':
        x = int(x.item())
    if x.ndim == 0 and x.dtype == 'float':
        x = float(x.item())
    return x