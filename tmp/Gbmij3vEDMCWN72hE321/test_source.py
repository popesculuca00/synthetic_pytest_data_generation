import torch
import source  # assuming the source code file is named 'source.py'

def test_timeseries_interpolate_batch_times():
    times1 = torch.tensor([[0, 2, 4, 6, 8, 10]])
    values1 = torch.tensor([[1, 2, 3, 4, 5, 6]])
    times2 = torch.tensor([[10, 12, 14, 16, 18, 20]])
    values2 = torch.tensor([[7, 8, 9, 10, 11, 12]])
    t = torch.tensor([5, 12])

    expected_output = torch.tensor([[3.5, 7.5]])

    assert torch.allclose(source.timeseries_interpolate_batch_times(times1, values1, t), expected_output), 'Test Case 1 Failed'
    assert torch.allclose(source.timeseries_interpolate_batch_times(times2, values2, t), expected_output), 'Test Case 2 Failed'