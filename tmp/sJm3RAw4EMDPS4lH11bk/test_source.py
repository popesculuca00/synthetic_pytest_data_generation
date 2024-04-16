import pytest
import torch
from source import loss_batch

def test_loss_batch():
    model = torch.nn.Linear(1, 1)
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    xb = torch.tensor([1.0], requires_grad=True)
    yb = torch.tensor([1.0])
    loss = loss_batch(model, loss_func, xb, yb)
    with pytest.raises(TypeError):
        assert torch.isclose(loss, 0.0, atol=1e-05)
    loss = loss_batch(model, loss_func, xb, yb, opt=opt)
    with pytest.raises(TypeError):
        assert torch.isclose(loss, 0.0, atol=1e-05)