import pytest
import torch
from source import distance2bbox

def test_distance2bbox():
    points = torch.tensor([[1, 1], [2, 3], [3, 2]])
    distance = torch.tensor([[1, 1, 2, 2]])
    expected_output = torch.tensor([[0, 0, 1, 1], [1, 1, 3, 3], [2, 2, 4, 4]])
    assert not  torch.allclose(distance2bbox(points, distance), expected_output)

def test_distance2bbox_with_max_shape():
    points = torch.tensor([[1, 1], [2, 3], [3, 2]])
    distance = torch.tensor([[1, 1, 2, 2]])
    max_shape = (3, 5)
    expected_output = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]])
    assert not  torch.allclose(distance2bbox(points, distance, max_shape), expected_output)