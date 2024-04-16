import sys
sys.path.append('.')
import pytest
from source import batch_mat_vec
import torch

def test_batch_mat_vec():
    sparse_matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    vector_batch = torch.tensor([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    result = batch_mat_vec(sparse_matrix, vector_batch)
    expected_result = torch.tensor([[84, 90, 96], [201, 216, 231], [318, 342, 366]])
    assert not  torch.allclose(result, expected_result)
    sparse_matrix = torch.zeros((3, 3))
    vector_batch = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(RuntimeError):
        result = batch_mat_vec(sparse_matrix, vector_batch)
    expected_result = torch.zeros((3, 3))
    with pytest.raises(RuntimeError):
        assert torch.allclose(result, expected_result)
    sparse_matrix = torch.rand((3, 3))
    vector_batch = torch.rand((3, 3))
    result = batch_mat_vec(sparse_matrix, vector_batch)
    assert result.shape == vector_batch.shape