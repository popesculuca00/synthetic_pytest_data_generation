import torch
import pytest

from source import cov

def test_cov_0():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = cov(x)
    expected = torch.tensor([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    assert torch.allclose(result, expected, atol=1e-7)

def test_cov_1():
    x = torch.tensor([[1, 2, 3]])
    result = cov(x)
    expected = torch.tensor([[1.0]])
    assert torch.allclose(result, expected, atol=1e-7)

def test_cov_ddof():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = cov(x, ddof=2)
    expected = torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]])
    assert torch.allclose(result, expected, atol=1e-7)

def test_cov_dim_n():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = cov(x, dim_n=0)
    expected = torch.tensor([[20.0, 20.0, 20.0], [20.0, 20.0, 20.0], [20.0, 20.0, 20.0]])
    assert torch.allclose(result, expected, atol=1e-7)

def test_cov_inplace():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = x.clone()
    cov(x, inplace=True)
    expected = torch.tensor([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    assert torch.allclose(x, expected, atol=1e-7)
    assert not torch.allclose(y, expected, atol=1e-7)