import torch
import pytest
from source import hard_examples_mining

def test_hard_examples_mining():
    # Create dummy tensors
    dist_mat = torch.rand(10, 10)
    identity_mat = torch.zeros(10, 10)
    # Fill the upper triangle of the identity matrix with ones
    torch.triu(identity_mat, -1)
    
    # Call the function and check if results are as expected
    dist_ap, dist_an, hard_positive_idxes, hard_negative_idxes = hard_examples_mining(dist_mat, identity_mat, return_idxes=True)
    assert dist_ap.shape == dist_an.shape == hard_positive_idxes.shape == hard_negative_idxes.shape == (10,)
    assert dist_ap.min() == dist_an.max() == 0
    assert dist_ap.max() > dist_an.min() > 0
    assert (hard_positive_idxes != hard_negative_idxes).all()

test_hard_examples_mining()