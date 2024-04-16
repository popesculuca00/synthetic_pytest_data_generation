# test_source.py
import pytest
import torch
from source import _normalize_images

def test_normalize_images():
    # Generate a random tensor
    images = torch.rand(3, 4, 5)

    # Call the function and get the result
    result = _normalize_images(images)

    # We only need to check if the type of the result is correct
    assert isinstance(result, torch.Tensor), "The function did not return a torch tensor"

    # You can add more asserts to check if the function works correctly, for example:
    # assert torch.allclose(result, expected_result), "The function did not normalize the images correctly"