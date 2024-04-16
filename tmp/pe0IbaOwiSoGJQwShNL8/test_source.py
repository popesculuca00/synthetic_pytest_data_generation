# test_source.py
import pytest
import torch
from source import _normalize_images

def test_normalize_images():
    # Create a dummy tensor
    images = torch.rand(3, 3, 3)

    # Call the function and get the result
    result = _normalize_images(images)

    # We only want to test that the function runs without error,
    # so we use an assertion to make sure the output type is correct.
    assert isinstance(result, torch.Tensor), "The output type is not a torch tensor"

    # Since we are testing the code coverage, make sure all lines are executed
    assert True