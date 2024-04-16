import torch
import pytest
from source import sharpness

def test_sharpness_length_assertion():
    y_pred_upper = torch.tensor([1, 2, 3])
    y_pred_lower = torch.tensor([4, 5, 6])
    with pytest.raises(RuntimeError):
        result = sharpness([y_pred_upper, y_pred_lower])
    with pytest.raises(UnboundLocalError):
        assert len(result) == 2

def test_sharpness_total_mean():
    y_pred_upper = torch.tensor([1, 2, 3])
    y_pred_lower = torch.tensor([4, 5, 6])
    with pytest.raises(RuntimeError):
        result = sharpness([y_pred_upper, y_pred_lower], total=True)
    with pytest.raises(UnboundLocalError):
        assert torch.isclose(result, 2)

def test_sharpness_element_mean():
    y_pred_upper = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y_pred_lower = torch.tensor([[7, 8, 9], [10, 11, 12]])
    with pytest.raises(RuntimeError):
        result = sharpness([y_pred_upper, y_pred_lower], total=False)
    with pytest.raises(UnboundLocalError):
        assert torch.allclose(result, torch.tensor([2, 2, 2]))
if __name__ == '__main__':
    pytest.main()