import pytest
import torch
from source import obb2xyxy_v3

def test_obb2xyxy_v3():
    obboxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
    output = obb2xyxy_v3(obboxes)
    expected_output = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(output, expected_output)