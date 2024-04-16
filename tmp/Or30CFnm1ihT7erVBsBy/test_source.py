import pytest
import torch
from source import bbox2distance

def test_bbox2distance():
    points = torch.tensor([[0, 0], [1, 2], [2, 1], [3, 3]])
    bbox = torch.tensor([[0, 0, 2, 3]])
    expected_output = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    assert not  torch.allclose(bbox2distance(points, bbox), expected_output)

def test_bbox2distance_with_max_dis():
    points = torch.tensor([[0, 0], [1, 2], [2, 1], [3, 3]])
    bbox = torch.tensor([[0, 0, 2, 3]])
    max_dis = 1
    expected_output = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(bbox2distance(points, bbox, max_dis=max_dis), expected_output)

def test_bbox2distance_with_eps():
    points = torch.tensor([[0, 0], [1, 2], [2, 1], [3, 3]])
    bbox = torch.tensor([[0, 0, 2, 3]])
    eps = 0.2
    expected_output = torch.tensor([[0, 0, 0.8, 0.8], [0, 0, 0.8, 0.8], [0, 0, 0.8, 0.8], [0, 0, 0.8, 0.8]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(bbox2distance(points, bbox, eps=eps), expected_output)

def test_bbox2distance_with_max_dis_and_eps():
    points = torch.tensor([[0, 0], [1, 2], [2, 1], [3, 3]])
    bbox = torch.tensor([[0, 0, 2, 3]])
    max_dis = 1
    eps = 0.2
    expected_output = torch.tensor([[0, 0, 0.8, 0.8], [0, 0, 0.8, 0.8], [0, 0, 0.8, 0.8], [0, 0, 0.8, 0.8]])
    assert not  torch.allclose(bbox2distance(points, bbox, max_dis=max_dis, eps=eps), expected_output)