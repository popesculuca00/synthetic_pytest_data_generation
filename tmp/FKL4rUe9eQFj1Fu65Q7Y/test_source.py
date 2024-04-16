import pytest
import torch
import numpy as np
import sys
sys.path.append(".")
import source

def test_rae():
    # Assuming y_true and y_pred are numpy arrays
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    # Converting to torch tensors
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    # Using the function from source script
    loss = source.rae(y_true_tensor, [y_pred_tensor])

    # Asserting that the output is as expected
    assert torch.isclose(loss, torch.tensor(0.0)), "The loss is not zero as expected"

# Running the test
test_rae()