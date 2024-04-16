def kl_normal_loss(mean, logvar, mean_dim=None):
    
    if mean_dim is None:
        mean_dim = [0]
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=mean_dim)
    return latent_kl