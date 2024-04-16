import torch

def max_sliced_wasserstein_distance(max_projected_true, max_projected_fake, device):
    

    # The input num_features can be considered as num_projections
    max_projected_true = max_projected_true.transpose(0, 1)
    max_projected_fake = max_projected_fake.transpose(0, 1)

    # Sort the max projection. If it has more than 1 component, sort by row.
    sorted_true = torch.sort(max_projected_true, dim=1)[0]
    sorted_fake = torch.sort(max_projected_fake, dim=1)[0]    

    # Get Wasserstein-2 distance
    return torch.pow(sorted_true - sorted_fake, 2).mean()