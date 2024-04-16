import torch

# importing the function from source file
from source import cross_squared_distance_matrix

class TestCrossSquaredDistanceMatrix:

    def test_function(self):
        # creating two random tensors
        x = torch.randn(1, 3)
        y = torch.randn(4, 3)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)
        y = y.to(device)

        # calling the function
        result = cross_squared_distance_matrix(x, y, device)

        # adding an assertion to test the output
        assert torch.allclose(result, torch.tensor([[0.3344, 0.3344, 0.3344, 0.3344]])), "The function did not return the expected output"