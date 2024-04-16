import sys
sys.path.append(".")
import source  # assuming the source code is in the same directory
import pytest
import torch

def test_train_batch():
    # initialize model, x, target, optimizer, and criterion
    model = torch.nn.Linear(1, 1)  # example model
    x = torch.tensor([[1.], [2.]])  # example input
    target = torch.tensor([[3.], [4.]])  # example target
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # example optimizer
    criterion = torch.nn.MSELoss()  # example criterion

    # call the function and get loss
    loss = source.train_batch(model, x, target, optimizer, criterion)

    # assert that the loss is not None
    assert loss is not None

if __name__ == "__main__":
    test_train_batch()