import sys
sys.path.append('.')
import source
import pytest
import torch

def test_bce_loss():
    """Testing the bce_loss function"""
    input_tensor = torch.tensor([1.0, -1.0, 0.0])
    target_tensor = torch.tensor([1.0, -1.0, 1.0])
    output = source.bce_loss(input_tensor, target_tensor)
    expected_output = torch.tensor([0.0, 0.0, 0.0])
    assert not  torch.allclose(output, expected_output)
pytest.main()