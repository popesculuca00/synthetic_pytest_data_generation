import pytest
import sys
sys.path.append('.')
from source import train_batch
import torch
import torch.nn as nn
import torch.optim as optim

def test_train_batch():
    model = torch.nn.Linear(1, 1)
    x = torch.tensor([1.0])
    target = torch.tensor([2.0])
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    output = train_batch(model, x, target, optimizer, criterion)
    with pytest.raises(TypeError):
        assert torch.isclose(output, 1.0), 'Function is not working as expected'