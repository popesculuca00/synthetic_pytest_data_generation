import pytest
import torch
from source import pixel_unshuffle  # import from the source.py file

def test_pixel_unshuffle():
    # create dummy input data
    x = torch.randn(2, 3, 10, 10)
    scale = 2

    # run the function
    output = pixel_unshuffle(x, scale)

    # perform assertion
    assert output.shape == (2, 3, 5, 5), "The output shape doesn't match the expected shape"