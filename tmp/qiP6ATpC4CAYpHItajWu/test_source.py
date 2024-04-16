import torch
import pytest

from source import distance2bbox

def test_distance2bbox():
    points = torch.tensor([[1, 1], [2, 3], [3, 2]])
    distance = torch.tensor([[1, 1, 2, 2]])
    max_shape = (3, 4)

    result = distance2bbox(points, distance, max_shape)

    expected_result = torch.tensor([[0, 0, 2, 2], 
                                    [0, 1, 2, 3], 
                                    [1, 0, 3, 2]])

    assert torch.allclose(result, expected_result)

test_distance2bbox()