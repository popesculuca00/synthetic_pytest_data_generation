import pytest
import torch

def test_transform_homogeneous():
    torch.manual_seed(42)  # for reproducibility
    # test case with random input data
    matrices = torch.rand(10, 3, 3)  # batch size 10, 3x3 matrices
    vertices = torch.rand(10, 3, 3)  # batch size 10, 3x3 vertices

    homogeneous_coord = torch.ones([vertices.shape[0], vertices.shape[1], 1]).to(vertices.device)
    vertices_homogeneous = torch.cat([vertices, homogeneous_coord], 2)

    result = torch.matmul(vertices_homogeneous, matrices.transpose(1, 2))
    expected_result = vertices_homogeneous @ matrices.transpose(1, 2)  # expected result using PyTorch's matmul

    assert torch.allclose(result, expected_result), 'The output does not match the expected result.'

# without a Pytest plugin, we need to manually call the test function
test_transform_homogeneous()