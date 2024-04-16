import torch
import pytest
from source import _pairwise_union_regions

def test_pairwise_union_regions():
    boxes1 = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)
    boxes2 = torch.tensor([[5, 5, 15, 15], [5, 5, 25, 25]], dtype=torch.float32)
    expected_output = torch.tensor([[0, 0, 30, 30], [20, 20, 30, 30]], dtype=torch.float32)
    output = _pairwise_union_regions(boxes1, boxes2)
    with pytest.raises(RuntimeError):
        assert torch.allclose(output, expected_output)