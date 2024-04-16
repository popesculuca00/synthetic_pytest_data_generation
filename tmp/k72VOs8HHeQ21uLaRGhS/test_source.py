import torch
import pytest

from source import pairwise_orthogonalization_torch

@pytest.fixture
def input_data():
    v1 = torch.rand(10, 3)
    v2 = torch.rand(10, 3)
    return v1, v2

def test_pairwise_orthogonalization_torch(input_data):
    v1, v2 = input_data
    v1_orth, EVR, EVR_total_weighted, EVR_total_unweighted = pairwise_orthogonalization_torch(v1, v2)
    assert torch.allclose(v1_orth.norm(dim=0), torch.zeros(10))  # The norm of the orthogonalized vector should be 0
    assert torch.isclose(EVR.mean(), torch.tensor(0.0))  # The average of EVR should be close to 0
    assert torch.isclose(EVR_total_weighted.mean(), torch.tensor(0.0))  # The average of EVR_total_weighted should be close to 0
    assert torch.isclose(EVR_total_unweighted, torch.tensor(0.0))  # EVR_total_unweighted should be close to 0