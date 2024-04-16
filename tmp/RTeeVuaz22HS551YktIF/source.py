import torch

def persistent_entropy(D, **kwargs):
    
    persistence = torch.diff(D)
    persistence = persistence[torch.isfinite(persistence)].abs()

    P = persistence.sum()
    probabilities = persistence / P

    # Ensures that a probability of zero will just result in
    # a logarithm of zero as well. This is required whenever
    # one deals with entropy calculations.
    indices = probabilities > 0
    log_prob = torch.zeros_like(probabilities)
    log_prob[indices] = torch.log2(probabilities[indices])

    return torch.sum(-probabilities * log_prob)