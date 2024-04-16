import pytest
from source import extract_odms  # assuming the function is in source.py

# Test the function with a random input of shape (2, 3, 4)
def test_extract_odms():
    voxelgrids = torch.randn(2, 3, 4)
    result = extract_odms(voxelgrids)
    assert result.shape == (2, 3, 4)  # Make sure the shape of the output is correct