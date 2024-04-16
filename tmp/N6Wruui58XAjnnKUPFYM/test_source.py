import torch
import pytest

# Import the source file
from source import normalize

# Test class to hold all the tests for the normalize function
class TestNormalize:
    def test_normalize(self):
        # Create a tensor
        tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        # Create the mean and std
        mean = [1, 2, 3]
        std = [2, 2, 2]
        
        # Call the normalize function
        normalized_tensor = normalize(tensor, mean, std)
        
        # Perform the assertion
        assert torch.allclose(normalized_tensor, torch.tensor([[[-1, 0, 1], [2, 3, 4]], [[5, 6, 7], [8, 9, 10]]]))

# Run the tests
pytest.main()