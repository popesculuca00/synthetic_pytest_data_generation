# test_source.py
import pytest
import torch
from source import linear  # assuming the function is defined in source.py

class TestLinear:
    def test_linear(self):
        # Given
        input = torch.tensor([1, 2, 3, 4])
        weight = torch.tensor([1, -1, 2, -2])
        bias = torch.tensor([1, 2, 3, 4])
        scale = 1.0
        zero_point = 0

        # When
        result = linear(input, weight, bias, scale, zero_point)

        # Then
        expected_result = torch.tensor([1, 0, 3, 2])
        assert torch.equal(result, expected_result), "Expected output does not match actual output"

if __name__ == "__main__":
    pytest.main()