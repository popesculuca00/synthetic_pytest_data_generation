import pytest
import torch
from source import bbox2distance

def test_bbox2distance():
    points = torch.tensor([[1, 1], [2, 2], [3, 3]])
    bbox = torch.tensor([[0, 0, 2, 2]])
    expected_output = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(bbox2distance(points, bbox), expected_output)
    points = torch.tensor([[0, 0], [1, 1], [2, 2]])
    bbox = torch.tensor([[0, 0, 1, 1]])
    expected_output = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0, 0, 0]])
    max_dis = 1.0
    with pytest.raises(RuntimeError):
        assert torch.allclose(bbox2distance(points, bbox, max_dis), expected_output)
    points = torch.tensor([[0, 0], [1, 1]])
    bbox = torch.tensor([[0, 0, 2, 2]])
    expected_output = torch.tensor([[0, 0, 1.1], [0, 0, 1.1]])
    eps = 1.0
    with pytest.raises(RuntimeError):
        assert torch.allclose(bbox2distance(points, bbox, eps=eps), expected_output)
    points = torch.tensor([[1, 1], [2, 2], [3, 3]])
    bbox = torch.tensor([[0, 0, 1, 1]])
    expected_output = torch.tensor([[0.5, 0.5, 1], [1, 1, 1], [1.1, 1.1, 1.1]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(bbox2distance(points, bbox), expected_output)