import pytest
import torch
from torch import nn, optim
from source import train_batch

def test_train_batch():
    model = nn.Linear(10, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(10, 10)
    target = torch.randint(0, 10, (10,))
    batch_loss, correct = train_batch(model, x, target, optimizer, criterion)
    assert isinstance(batch_loss, torch.Tensor), 'The function did not return a Torch Tensor'
    assert not  isinstance(correct, int), 'The function did not return an integer'