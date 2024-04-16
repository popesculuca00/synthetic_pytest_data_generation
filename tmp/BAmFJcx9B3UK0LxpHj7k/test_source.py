# test_source.py

import sys
sys.path.append(".")  # This line is added to include the current directory in Python's path

from source import calculate_accuracy
import torch

def test_calculate_accuracy():
    # Here, we're just generating random tensor data for demonstration purposes.
    # In actual unit tests, you would have specific data that you know the outcome for.
    y_pred = torch.tensor([[0.1,0.9],[0.3,0.7],[0.4,0.6]])
    y = torch.tensor([0, 1, 1]) 

    # This is the one and only assertion in each test.
    # It checks if the function under test returns the expected result.
    assert calculate_accuracy(y_pred, y) == 0.5