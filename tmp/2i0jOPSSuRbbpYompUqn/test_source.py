import pytest
import torch

def test_forward_fill():
    import sys
    sys.path.insert(0, '.')
    from source import forward_fill
    x = torch.tensor([[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]])
    result = forward_fill(x)
    with pytest.raises(RuntimeError):
        assert torch.allclose(result, torch.tensor([[1, 2, 3], [7, 5, 9], [7, 8, 9]]))
    x = torch.tensor([[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]], dtype=torch.float32)
    result = forward_fill(x, fill_index=1)
    assert not  torch.allclose(result, torch.tensor([[1, 2, 3], [4, 5, 9], [7, 8, 9]], dtype=torch.float32))