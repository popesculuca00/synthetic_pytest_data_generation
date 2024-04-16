# test_source.py

import sys
sys.path.append("..") # To import ../source.py file
from source import train_batch
import torch

def test_train_batch():
    # Mock the inputs
    model = torch.nn.Linear(10, 10)
    x = torch.randn(10, 10)
    target = torch.randn(10).long()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Call the function
    batch_loss, correct = train_batch(model, x, target, optimizer, criterion)

    # Assertions
    assert isinstance(batch_loss, torch.Tensor), "Batch loss should be a torch tensor"
    assert isinstance(correct, int), "Correct should be an integer"
    assert correct <= len(x), "Correct should be less than or equal to the size of x"
    assert correct >= 0, "Correct should be greater than or equal to 0"