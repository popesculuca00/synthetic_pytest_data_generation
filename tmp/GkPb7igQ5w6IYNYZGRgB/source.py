def _PointAdaptive_kNN(distances, indices, k_max=1000, D_thr=23.92812698, dim=None):
    r
    from Pipeline import _PAk

    # The adaptive k-Nearest Neighbor density estimator
    k_hat, dc, densities, err_densities = _PAk.get_densities(dim, distances, k_max, D_thr, indices)

    return densities, err_densities, k_hat, dc