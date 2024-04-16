import torch
import pytest

def test_generate_transformation_matrix():
    camera_position = torch.tensor([1.0, 2.0, 3.0])
    look_at = torch.tensor([0.0, 0.0, 0.0])
    camera_up_direction = torch.tensor([0.0, 1.0, 0.0])
    expected_output = torch.tensor([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -2.0], [0.0, 0.0, 1.0, -3.0], [0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(generate_transformation_matrix(camera_position, look_at, camera_up_direction), expected_output)