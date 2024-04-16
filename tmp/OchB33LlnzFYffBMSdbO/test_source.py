import torch
import pytest
import os

# Import the source code
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
import source 

def test_sphere_distance_torch():
    # Create test data
    x1 = torch.randn(5, 10)
    x2 = torch.randn(5, 10)

    # Compute the distance
    result = source.sphere_distance_torch(x1, x2)

    # This is your single assertion, change it to match your needs
    assert torch.allclose(result, torch.acos(torch.bmm(x1.view(-1, 1, x1.shape[-1]), x2.view(-1, x2.shape[-2], 1)).view(x1.shape[:-2])))

def test_sphere_distance_torch_diag():
    # Create test data
    x1 = torch.randn(5, 10)
    x2 = torch.randn(5, 10)

    # Compute the distance
    result = source.sphere_distance_torch(x1, x2, diag=True)

    # This is your single assertion, change it to match your needs
    assert torch.allclose(result, torch.acos(torch.bmm(x1.unsqueeze(-1).transpose(-1, -2), x2.unsqueeze(-1)).squeeze(-1)))