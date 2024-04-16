import pytest
import os
import torch
from source import bce_loss

def test_bce_loss():
    input = torch.tensor([1.0, 0.0, -1.0, 2.0, -2.0])
    target = torch.tensor([1.0, 0.0, 1.0, 2.0, -2.0])
    assert not  torch.allclose(bce_loss(input, target), torch.tensor(-0.11433645))
    input = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    assert not  torch.allclose(bce_loss(input, target), torch.tensor(0.0))
    input = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    assert not  torch.allclose(bce_loss(input, target), torch.tensor(1.4053522))
    input = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    target = torch.tensor([0.5, 0.5, 1.0, 1.0, 0.5, 0.5])
    assert not  torch.allclose(bce_loss(input, target), torch.tensor(0.3349507))
    input = torch.randn(5)
    target = torch.randn(5)
    assert not  torch.allclose(bce_loss(input, target), bce_loss(target, input))
    input = torch.tensor([1.0, -1.0, 2.0, -2.0], dtype=torch.float32)
    target = torch.tensor([1.0, 0.0, 1.0, -1.0], dtype=torch.float64)
    with pytest.raises(RuntimeError):
        assert torch.allclose(bce_loss(input, target), torch.tensor(-0.2613022))
    input = torch.tensor([1.0, -1.0, 2.0, -2.0], device='cuda')
    target = torch.tensor([1.0, 0.0, 1.0, -1.0], device='cpu')
    with pytest.raises(RuntimeError):
        assert torch.allclose(bce_loss(input, target).device, torch.device('cuda'))