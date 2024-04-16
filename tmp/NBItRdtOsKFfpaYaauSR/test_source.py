import torch
import pytest

from source import get_backward_gradient  # assuming the function is in source.py

def test_get_backward_gradient():
    pred_y = torch.rand((2, 3, 4, 5))
    y = [1, 2, 3]
    
    # Test with pred_y being a torch tensor and y being a list
    grad = get_backward_gradient(pred_y, y)
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == pred_y.shape == (2, 3, 4, 5)
    assert torch.allclose(grad[:, :, 1, 1], torch.tensor([[1., 1., 1.], [1., 1., 1.]]))

    # Test with pred_y being a torch tensor and y being a torch tensor
    y = torch.tensor([1, 2, 3], device=pred_y.device)
    grad = get_backward_gradient(pred_y, y)
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == pred_y.shape == (2, 3, 4, 5)
    assert torch.allclose(grad[:, :, 1, 1], torch.tensor([[1., 1., 1.], [1., 1., 1.]]))

    # Test with pred_y being a torch tensor and y being a different tensor
    y = torch.rand((2, 3, 4, 5), dtype=torch.long, device=pred_y.device)
    grad = get_backward_gradient(pred_y, y)
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == pred_y.shape == y.shape
    assert torch.allclose(grad, torch.ones_like(pred_y))

# Running the test
pytest.main()