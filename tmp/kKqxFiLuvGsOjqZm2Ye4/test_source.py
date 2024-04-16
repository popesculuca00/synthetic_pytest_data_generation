import pytest
import torch
from source import categorical_focal_loss

def test_categorical_focal_loss():
    y_true = torch.tensor([[0, 1, 0], [0, 0, 1]])
    y_pred = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    loss = categorical_focal_loss(y_true, y_pred)
    with pytest.raises(TypeError):
        assert torch.isclose(loss, 0.21212121212121213)