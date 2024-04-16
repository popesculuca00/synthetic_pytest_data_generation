import torch

def compute_pairwise_distances(x, y):
    

    if not len(x.size()) == len(y.size()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size()[1] != y.size()[1]:
        raise ValueError('The number of features should be the same.')

    # By making the `inner' dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    norm = lambda x: torch.sum(x * x, 1)
    return norm(x.unsqueeze(2) - y.t())