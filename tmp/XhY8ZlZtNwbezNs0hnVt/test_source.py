# test_source.py
import pytest
from source import compute_accuracy
import torch

def test_compute_accuracy():
    # Create mock y tensor
    y = torch.randn(10, 5)
    
    # Create mock outputs dictionary
    outputs = {
        "lm_scores": torch.randn(10, 10, 100),
        "span_b_scores": torch.randn(10, 100),
        "span_e_scores": torch.randn(10, 100),
    }

    # Call the function and get the expected result
    expected_result = compute_accuracy(outputs, y)

    # Now we will create some dummy data to check if it works with that
    y_dummy = torch.randn(10, 6)
    outputs_dummy = {
        "lm_scores": torch.randn(10, 100),
        "span_b_scores": torch.randn(100),
        "span_e_scores": torch.randn(100),
    }

    # Call the function with dummy data and compare the result with the expected result
    result_dummy = compute_accuracy(outputs_dummy, y_dummy)
    assert expected_result == result_dummy