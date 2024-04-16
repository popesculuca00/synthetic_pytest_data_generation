import pytest
import torch
from source import focal_loss  # replace with the actual path to your source file

def test_focal_loss():
    output = torch.tensor([0.7, 0.2, 0.9])
    target = torch.tensor([1., 0., 1.])
    assert torch.abs(focal_loss(output, target) - 0.094317) < 1e-5

test_focal_loss()