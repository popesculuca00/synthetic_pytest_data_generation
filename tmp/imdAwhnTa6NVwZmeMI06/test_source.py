import pytest
import torch
from source import distance2bbox

def test_distance2bbox():
    points = torch.tensor([[0, 0], [1, 2], [3, 4]])
    distance = torch.tensor([[1, 1, 2, 2]])
    max_shape = (4, 6)
    expected_output = torch.tensor([[ -1., -1., 0., 0.],
                                    [ 0., 0., 2., 2.],
                                    [ 1., 1., 3., 3.]])
    assert torch.allclose(distance2bbox(points, distance, max_shape), expected_output)

test_distance2bbox()