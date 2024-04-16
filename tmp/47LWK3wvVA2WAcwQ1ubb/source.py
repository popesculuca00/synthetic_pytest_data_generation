import torch

def my_sample_permutations(n_permutations, n_objects):
    
    random_pre_perm = torch.empty(n_permutations, n_objects).uniform_(0, 1)
    _, permutations = torch.topk(random_pre_perm, k = n_objects)
    return permutations