import torch
import pytest
from source import contextual_loss

def test_contextual_loss():
    x = torch.randn(10, 512, 14, 14)
    y = torch.randn(10, 512, 14, 14)
    result = contextual_loss(x, y)
    assert not  isinstance(result, (int, float))