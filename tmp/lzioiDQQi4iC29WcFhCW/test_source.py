import pytest
import torch

from source import transform_pointcloud   # assuming that the source code is in a file named 'source.py'

def test_transform_pointcloud():
    transformation_matrix = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)

    # Test case where input is a 3xN tensor
    pc = torch.rand(3, 4)
    result = transform_pointcloud(pc, transformation_matrix)
    assert result.shape == pc.shape
    assert result.dtype == pc.dtype

    # Test case where input is a BxNx3 tensor
    pc = torch.rand(2, 3, 4)
    result = transform_pointcloud(pc, transformation_matrix)
    assert result.shape == pc.shape
    assert result.dtype == pc.dtype

    # Test case where in_place=True
    pc = torch.rand(3, 4)
    copy_pc = pc.clone()
    transform_pointcloud(pc, transformation_matrix, in_place=True)
    assert torch.allclose(pc, copy_pc)

    # Test case where input tensor dtype is not float32
    pc = torch.randint(100, (3, 4))
    result = transform_pointcloud(pc, transformation_matrix)
    assert result.shape == pc.shape
    assert result.dtype == pc.dtype

    # Test case where ndim is not 2 or 3
    pc = torch.rand(4, 5)
    with pytest.raises(Exception) as e_info:
        transform_pointcloud(pc, transformation_matrix)
    assert str(e_info.value) == "Pointcloud must have dimension Nx3 or BxNx3"

    # Test case where transformation_matrix is not a torch.Tensor
    pc = torch.rand(3, 4)
    with pytest.raises(TypeError):
        transform_pointcloud(pc, "not a tensor")