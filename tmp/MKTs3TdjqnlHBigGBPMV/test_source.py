import pytest
import torch
from source import sequence_mask

def test_sequence_mask():
    seq_ids = torch.tensor([1, 2, 3, 4])
    valid_lengths = torch.tensor([4, 3, 2, 1])
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(sequence_mask(seq_ids, valid_lengths), expected)
    seq_ids = torch.tensor([5, 4, 3, 2])
    valid_lengths = torch.tensor([1, 2, 3, 4])
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(sequence_mask(seq_ids, valid_lengths), expected)
    seq_ids = torch.tensor([1, 2, 3])
    valid_lengths = torch.tensor([1, 2, 3, 4])
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(sequence_mask(seq_ids, valid_lengths), expected)
    seq_ids = torch.tensor([4, 4, 4, 4])
    valid_lengths = torch.tensor([4, 4, 4, 4])
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(sequence_mask(seq_ids, valid_lengths), expected)
    seq_ids = torch.tensor([5, 4, 3, 2])
    valid_lengths = torch.tensor([1, 2, 3, 4])
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(sequence_mask(seq_ids, valid_lengths), expected)
    seq_ids = torch.tensor([])
    valid_lengths = torch.tensor([])
    expected = torch.tensor([])
    with pytest.raises(RuntimeError):
        assert torch.allclose(sequence_mask(seq_ids, valid_lengths), expected)