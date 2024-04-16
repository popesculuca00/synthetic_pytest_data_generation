import pytest
from source import train_batch
import torch

def test_train_batch():
    model = torch.nn.Linear(1, 1)
    x = torch.tensor([1.0])
    target = torch.tensor([2.0])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    result = train_batch(model, x, target, optimizer, criterion)
    assert result == 2.2025132179260254, 'The function did not produce the expected result'