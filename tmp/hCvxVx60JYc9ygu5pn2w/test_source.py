# test_source.py
import torch
import pytest

from source import deptree_nonproj

def test_deptree_nonproj():
    # Create random test data
    arc_scores = torch.randn(4, 5)
    eps = 1e-5

    # Calculate expected output
    expected_output = deptree_nonproj(arc_scores, eps)

    # Calculate actual output
    actual_output = deptree_nonproj(arc_scores, eps)

    # Check if the outputs match
    assert torch.allclose(expected_output, actual_output)

if __name__ == "__main__":
    pytest.main()