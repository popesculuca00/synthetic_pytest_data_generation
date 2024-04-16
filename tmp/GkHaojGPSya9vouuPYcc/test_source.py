import pytest
import torch
from source import is_complex_symmetric

def test_is_complex_symmetric():
    z = torch.tensor([[1.0 + 2j, 3j], [4 + 5j, 6j]])
    assert not  is_complex_symmetric(z)