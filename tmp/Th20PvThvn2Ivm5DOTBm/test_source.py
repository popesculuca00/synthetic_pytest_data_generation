import sys
sys.path.append(".")  # This line is to include the current directory, where source.py is, in the path
from source import train_batch  # Here we are importing the train_batch function from source.py
import torch  # We need torch for this test

def test_train_batch():
    # We will create some random data for testing
    model = torch.nn.Linear(1, 1)  # A simple linear model
    x = torch.tensor([1.0])  # Input
    target = torch.tensor([2.0])  # Corresponding target value
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # An optimizer
    criterion = torch.nn.MSELoss()  # The loss function

    # Call the function with the randomly generated data
    batch_loss = train_batch(model, x, target, optimizer, criterion)

    # We will check if the returned loss is a number (float)
    assert isinstance(batch_loss, float), "The function should return a float"