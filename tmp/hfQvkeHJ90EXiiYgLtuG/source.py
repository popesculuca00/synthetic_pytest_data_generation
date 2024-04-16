def gather_along_dim_with_dim_single(x, target_dim, source_dim, indices):
    
    assert len(indices.shape) == 1
    assert x.shape[source_dim] == indices.shape[0]

    # unsqueeze in all dimensions except the source dimension
    new_shape = [1] * x.ndimension()
    new_shape[source_dim] = -1
    indices = indices.reshape(*new_shape)

    # repeat in all dimensions - but preserve shape of source dimension,
    # and make sure target_dimension has singleton dimension
    expand_shape = list(x.shape)
    expand_shape[source_dim] = -1
    expand_shape[target_dim] = 1
    indices = indices.expand(*expand_shape)

    out = x.gather(dim=target_dim, index=indices)
    return out.squeeze(target_dim)