import torch

def cov(x, ddof=1, dim_n=1, inplace=False):
    
    if len(x.shape) != 2:
        raise ValueError('The function supports only 2D matrices')
    if dim_n not in {0, 1}:
        raise ValueError('dim_n must be either 0 or 1')

    # Center the data on the mean.
    if dim_n == 1:
        keepdim = True
    else:
        keepdim = False
    mean = torch.mean(x, dim_n, keepdim=keepdim)
    if inplace:
        x -= mean
    else:
        x = x - mean

    # Average normalization factor.
    n = x.shape[dim_n] - ddof

    # Compute the covariance matrix
    if dim_n == 0:
        c = x.t().matmul(x) / n
    else:
        c = x.matmul(x.t()) / n

    return c