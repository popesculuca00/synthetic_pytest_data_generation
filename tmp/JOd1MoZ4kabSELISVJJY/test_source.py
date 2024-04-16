import pytest
import torch
from source import one_hot_encode

def test_one_hot_encode_0D_tensor():
    tensor = torch.tensor(3)
    num_classes = 5
    expected_output = torch.tensor([1, 0, 0, 0, 0])
    with pytest.raises(RuntimeError):
        assert torch.allclose(one_hot_encode(tensor, num_classes), expected_output)

def test_one_hot_encode_1D_tensor():
    tensor = torch.tensor([3, 1, 4])
    num_classes = 5
    expected_output = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(one_hot_encode(tensor, num_classes), expected_output)

def test_one_hot_encode_2D_tensor():
    tensor = torch.tensor([[3, 1], [4, 2]])
    num_classes = 5
    expected_output = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(one_hot_encode(tensor, num_classes), expected_output)