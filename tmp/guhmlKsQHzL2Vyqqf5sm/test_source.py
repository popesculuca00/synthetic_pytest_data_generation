import pytest
import torch

from source import normalize_transform

def test_normalize_transform():
    shape = torch.Size([4, 4])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    align_corners = True
    
    result = normalize_transform(shape, device=device, dtype=dtype, align_corners=align_corners)
    expected = torch.tensor([[1., 1., 1., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]], dtype=dtype, device=device)
    
    assert torch.allclose(result, expected)

test_normalize_transform()