import pytest
import torch
from source import bbox2distance

def test_bbox2distance():
    points = torch.tensor([[1, 2], [3, 4], [5, 6]])
    bbox = torch.tensor([[0, 0, 2, 3]])
    expected_output = torch.tensor([[1, 2, 0, 1], [2, 3, 0, 1], [4, 5, 0, 1]])
    assert not  torch.allclose(bbox2distance(points, bbox), expected_output)

def test_bbox2distance_with_max_dis():
    points = torch.tensor([[1, 2], [3, 4], [5, 6]])
    bbox = torch.tensor([[0, 0, 2, 3]])
    max_dis = 1
    expected_output = torch.tensor([[1, 2, 0, 1], [2, 3, 0, 1], [1, 1, 0, 1]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(bbox2distance(points, bbox, max_dis), expected_output)

def test_bbox2distance_with_eps():
    points = torch.tensor([[1, 2], [3, 4], [5, 6]])
    bbox = torch.tensor([[0, 0, 2, 3]])
    eps = 0.2
    expected_output = torch.tensor([[0.8, 1, 0, 0.8], [1, 1, 0, 0.8], [1.2, 1.2, 0, 0.8]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(bbox2distance(points, bbox, eps=eps), expected_output)

def test_bbox2distance_with_max_dis_and_eps():
    points = torch.tensor([[1, 2], [3, 4], [5, 6]])
    bbox = torch.tensor([[0, 0, 2, 3]])
    max_dis = 1
    eps = 0.2
    expected_output = torch.tensor([[0.8, 1, 0, 0.8], [1, 1, 0, 0.8], [0.8, 0.8, 0, 0.8]])
    assert not  torch.allclose(bbox2distance(points, bbox, max_dis, eps), expected_output)