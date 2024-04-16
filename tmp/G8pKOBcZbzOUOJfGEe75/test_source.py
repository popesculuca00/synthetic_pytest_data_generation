import sys
sys.path.insert(0, '..') # to import the module from the parent directory
import pytest
from source import btdot
import torch

def test_btdot():
    large = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
    small = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.tensor([[5., 7., 9.], [11., 13., 15.], [17., 19., 21.], [23., 25., 27.]])
    assert torch.allclose(btdot(large, small), expected, atol=1e-6)

test_btdot()