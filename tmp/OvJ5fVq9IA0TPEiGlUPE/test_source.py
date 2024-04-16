import torch
import pytest
from source import invert_permutation

def test_invert_permutation():
    perm = torch.randperm(10, dtype=torch.long)
    iperm = invert_permutation(perm)
    assert torch.allclose(iperm, torch.argsort(perm))
    perm = torch.tensor([[0, 1, 9], [1, 0, 2], [9, 2, 0]])
    with pytest.raises(RuntimeError):
        iperm = invert_permutation(perm)
    with pytest.raises(RuntimeError):
        assert torch.allclose(iperm, torch.tensor([[0, 1, 2], [1, 0, 9], [2, 9, 0]]))
    perm = torch.randperm(10000, dtype=torch.long)
    iperm = invert_permutation(perm)
    assert torch.allclose(iperm, torch.argsort(perm))
    perm = torch.randperm(10, dtype=torch.float)
    with pytest.raises(RuntimeError):
        iperm = invert_permutation(perm)
    with pytest.raises(RuntimeError):
        assert torch.allclose(iperm, torch.argsort(perm))
    perm = torch.randperm(10, dtype=torch.long).cuda()
    iperm = invert_permutation(perm)
    assert torch.allclose(iperm, torch.argsort(perm))
    perm = torch.empty(0, dtype=torch.long)
    with pytest.raises(RuntimeError):
        iperm = invert_permutation(perm)
    with pytest.raises(RuntimeError):
        assert torch.allclose(iperm, perm)
    with pytest.raises(TypeError):
        perm = torch.randperm(10, 10, dtype=torch.long)
    with pytest.raises(RuntimeError):
        iperm = invert_permutation(perm)
    with pytest.raises(RuntimeError):
        assert torch.allclose(iperm, torch.argsort(perm, dim=-1).T)
    with pytest.raises(TypeError):
        perm = torch.randperm(10, 10, 10, dtype=torch.long)
    with pytest.raises(RuntimeError):
        iperm = invert_permutation(perm)
    with pytest.raises(RuntimeError):
        assert torch.allclose(iperm, torch.argsort(perm, dim=-1).T)