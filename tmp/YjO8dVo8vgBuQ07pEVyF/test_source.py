from source import *
import pytest
import torch
from source import heatmap

def test_heatmap():
    """Test that the heatmap function returns expected output."""
    xdelta = 0.0
    ydelta = 0.0
    scale = 10.0
    precision = 15.0
    heatmap_dim = 19
    with pytest.raises(NameError):
        expected_output = torch.exp(-1 * precision * ((grid - xdelta) ** 2 + (grid - ydelta) ** 2))
    output = heatmap(xdelta, ydelta, scale, precision, heatmap_dim)
    with pytest.raises(UnboundLocalError):
        assert torch.allclose(output, expected_output), 'Output does not match expected result.'
if __name__ == '__main__':
    test_heatmap()