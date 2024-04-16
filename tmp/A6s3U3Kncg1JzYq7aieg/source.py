def repeat_batch(t, K, dim=0):
    
    shape = t.shape
    tiling = [1] * (len(shape) + 1)
    tiling[dim + 1] = K
    tiled = t.unsqueeze(dim + 1).repeat(tiling)
    old_bsz = shape[dim]
    new_bsz = old_bsz * K
    new_shape = list(shape[:dim]) + [new_bsz] + list(shape[dim+1:])
    return tiled.view(new_shape)