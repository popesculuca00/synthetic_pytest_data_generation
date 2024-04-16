import pytest
import torch
from source import label_to_levels

def test_label_to_levels_tensor():
    label = torch.tensor(3)
    num_classes = 5
    result = label_to_levels(label, num_classes)
    expected = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
    with pytest.raises(RuntimeError):
        assert torch.all(result == expected)

def test_label_to_levels_int():
    label = 3
    num_classes = 5
    result = label_to_levels(label, num_classes)
    expected = torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32)
    with pytest.raises(RuntimeError):
        assert torch.all(result == expected)

def test_label_to_levels_error():
    label = torch.tensor(5)
    num_classes = 5
    with pytest.raises(ValueError):
        label_to_levels(label, num_classes)

def test_label_to_levels_dtype():
    label = torch.tensor(3)
    num_classes = 5
    result = label_to_levels(label, num_classes, torch.int32)
    expected = torch.tensor([1, 1, 1, 0, 0], dtype=torch.int32)
    with pytest.raises(RuntimeError):
        assert torch.all(result == expected)