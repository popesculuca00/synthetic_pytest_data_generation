import torch
import pytest

from source import draw_shape

def test_draw_shape():
    pos = torch.tensor([[0.5, 0.5]], device="cuda")
    sigma_x = torch.tensor([0.5], device="cuda")
    sigma_y = torch.tensor([0.5], device="cuda")
    angle = torch.tensor([1.5708], device="cuda")   # pi/2 in radians
    size = 10

    output = draw_shape(pos, sigma_x, sigma_y, angle, size)

    assert output.shape == (1, 1, size, size), "Output shape does not match expected shape!"