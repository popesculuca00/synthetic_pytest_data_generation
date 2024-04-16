import pytest
import torch
from source import attention_padding_mask

def test_attention_padding_mask():
    q = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    k = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 0]])
    mask = attention_padding_mask(q, k, padding_index=0)
    with pytest.raises(RuntimeError):
        assert torch.allclose(mask, torch.tensor([[False, False, False, False], [False, False, False, True]]))