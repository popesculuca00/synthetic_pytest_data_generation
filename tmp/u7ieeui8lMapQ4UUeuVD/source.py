def _symmetrized_kl(dist1, dist2, eps=1e-8):
    

    num_dims = int(dist1.shape[-1] / 2)

    dist1_mean = dist1[..., :num_dims].unsqueeze(-3)
    dist1_logvar = dist1[..., num_dims:].unsqueeze(-3)
    dist1_var = eps + dist1_logvar.exp()

    dist2_mean = dist2[..., :num_dims].unsqueeze(-2)
    dist2_logvar = dist2[..., num_dims:].unsqueeze(-2)
    dist2_var = eps + dist2_logvar.exp()

    var_ratio12 = dist1_var / dist2_var
    # log_var_ratio12 = var_ratio12.log()
    # note that the log variance ratio cancels because of the summed KL.
    loc_sqdiffs = (dist1_mean - dist2_mean).pow(2)
    kl1 = 0.5 * (var_ratio12 + loc_sqdiffs / dist2_var - 1)
    kl2 = 0.5 * (var_ratio12.reciprocal() + loc_sqdiffs / dist1_var - 1)
    symmetrized_kl = kl1 + kl2
    return symmetrized_kl.sum(-1).transpose(-1, -2)