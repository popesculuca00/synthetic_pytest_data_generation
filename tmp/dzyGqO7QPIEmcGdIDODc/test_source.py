# test_source.py
import pytest
from source import dice_loss
import torch

def test_dice_loss_function():
    # We will use random tensors for inputs and targets
    inputs = torch.randn(10, 10)
    targets = torch.randn(10, 10)
    num_boxes = 100

    # Call the function and store the output
    output = dice_loss(inputs, targets, num_boxes)

    # We will use a very simple assertion to check if the output is a tensor
    # Notice that Pytest will automatically fail the test if the assertion fails
    assert isinstance(output, torch.Tensor), "The output is not a torch tensor"