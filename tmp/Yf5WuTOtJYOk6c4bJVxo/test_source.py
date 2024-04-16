import pytest
from source import train_batch
from torch.optim import SGD
import torch.nn.functional as F
import torch

def test_train_batch():
    model = torch.nn.Linear(1, 1)
    x = torch.tensor([[0.0], [1.0]])
    target = torch.tensor([[0.0], [1.0]])
    optimizer = SGD(model.parameters(), lr=0.1)
    criterion = F.mse_loss
    result = train_batch(model, x, target, optimizer, criterion)
    assert result == 1.4584650993347168, 'The function did not return the expected result.'