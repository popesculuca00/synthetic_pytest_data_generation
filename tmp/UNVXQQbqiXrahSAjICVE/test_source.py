import pytest
from source import dropout_mask
import torch

def test_dropout_mask():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    sz = (2, 3)
    dropout = 0.5
    mask = dropout_mask(x, sz, dropout)
    assert torch.allclose(mask, torch.tensor([[0, 1, 0], [1, 0, 1]]))