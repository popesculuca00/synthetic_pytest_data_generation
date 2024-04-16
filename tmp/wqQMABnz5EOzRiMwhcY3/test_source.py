import pytest
import torch

from source import square_distance

class TestSquareDistance:

    def test_square_distance(self):
        xyz1 = torch.rand((10, 3, 3))
        xyz2 = torch.rand((10, 3, 3))

        result = square_distance(xyz1, xyz2)

        assert result.shape == xyz1.shape, "The shape of the output does not match the input"
        assert not torch.isnan(result).any(), "The output contains nan values"
        assert not torch.isinf(result).any(), "The output contains inf values"

if __name__ == "__main__":
    pytest.main()