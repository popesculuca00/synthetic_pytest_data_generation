import pytest
import torch
from source import tangent_vectors

def test_tangent_vectors():
    normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    expected_output = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    output = tangent_vectors(normals)
    with pytest.raises(RuntimeError):
        assert torch.allclose(output, expected_output, atol=1e-06)