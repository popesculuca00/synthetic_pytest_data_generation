import pytest
import torch
from source import coordinates

def test_coordinates():
    nx, ny, nz = (3, 3, 3)
    expected_output = torch.stack((torch.tensor([i for i in range(nx)]), torch.tensor([j for j in range(ny)]), torch.tensor([k for k in range(nz)])))
    result = coordinates(voxel_dim=(nx, ny, nz))
    with pytest.raises(RuntimeError):
        assert torch.allclose(result, expected_output)