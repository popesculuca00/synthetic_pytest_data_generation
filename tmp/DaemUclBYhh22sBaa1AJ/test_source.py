# test_source.py
import pytest
import torch
from source import azimuthal_average  # assuming the function is in source.py

def test_azimuthal_average():
    # Test with an example image, assuming it is in the same directory
    image = torch.ones((10, 10))
    result = azimuthal_average(image)
    assert isinstance(result, torch.Tensor), "The function should return a torch tensor"
    assert result.shape == (1, 1), "The shape of the output tensor is incorrect"

if __name__ == "__main__":
    test_azimuthal_average()