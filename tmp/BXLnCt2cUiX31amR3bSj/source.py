import torch

def _compute_ece(prob, bin_mean_prob):
    
    pz_given_b = prob / torch.unsqueeze(torch.sum(prob, dim=0), 0)
    prob_correct = prob[1, :] / torch.sum(prob[1, :])
    ece = torch.sum(prob_correct * torch.abs(pz_given_b[1, :] - bin_mean_prob))

    return ece