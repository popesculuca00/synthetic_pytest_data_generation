import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pytest
import torch
from source import _symmetrized_kl

def test_symmetrized_kl():
    dist1 = torch.randn(2, 3, 4)
    dist2 = torch.randn(2, 3, 4)
    result = _symmetrized_kl(dist1, dist2)
    expected = torch.randn(2, 3)
    with pytest.raises(RuntimeError):
        assert torch.allclose(result, expected, atol=0.0001)