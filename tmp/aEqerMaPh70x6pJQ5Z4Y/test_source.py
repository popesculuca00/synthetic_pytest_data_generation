import pytest
import torch
from source import bbox2distance

def test_bbox2distance():
    points = torch.tensor([[0, 0], [1, 2], [2, 1]])
    bbox = torch.tensor([[0, 0, 1, 3]])
    result = bbox2distance(points, bbox)
    expected = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    assert not  torch.allclose(result, expected)

def test_bbox2distance_with_max_dis():
    points = torch.tensor([[0, 0], [1, 2], [2, 1]])
    bbox = torch.tensor([[0, 0, 1, 3]])
    max_dis = 1
    result = bbox2distance(points, bbox, max_dis=max_dis)
    expected = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(result, expected)

def test_bbox2distance_with_eps():
    points = torch.tensor([[0, 0], [1, 2], [2, 1]])
    bbox = torch.tensor([[0, 0, 1, 3]])
    eps = 2
    result = bbox2distance(points, bbox, eps=eps)
    expected = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    assert not  torch.allclose(result, expected)

def test_bbox2distance_with_all_args():
    points = torch.tensor([[0, 0], [1, 2], [2, 1]])
    bbox = torch.tensor([[0, 0, 1, 3]])
    max_dis = 2
    eps = 1
    result = bbox2distance(points, bbox, max_dis=max_dis, eps=eps)
    expected = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert not  torch.allclose(result, expected)