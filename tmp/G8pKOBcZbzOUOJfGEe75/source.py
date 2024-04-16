def btdot(large, small):
    
    dim_diff = large.dim() - small.dim()
    batch_dim = small.size(0)
    extra_dims = [1] * dim_diff
    remaining_dims = small.size()[1:]
    sview = small.view(batch_dim, *extra_dims, *remaining_dims)
    return (large * sview).sum(tuple(range(dim_diff + 1, large.dim())))