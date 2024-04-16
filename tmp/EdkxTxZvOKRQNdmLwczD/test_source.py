# Import the module for testing
import source  # replace "source" with the actual name of your python file

def test_grid_cluster():
    # Test the grid_cluster function
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    size = 2
    expected_output = torch.tensor([1, 3], dtype=torch.int32)
    assert torch.equal(source.grid_cluster(x, size), expected_output)  # make sure the output matches the expected output