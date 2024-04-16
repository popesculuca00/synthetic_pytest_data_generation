import pytest
import torch
from source import calc_square_dist

def test_calc_square_dist():
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
    y = torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], [[8.0, 9.0, 10.0], [11.0, 12.0, 13.0]]])
    result = calc_square_dist(x, y, True)
    expected = torch.tensor([[[5.0, 4.0, 3.0], [8.0, 5.0, 6.0]], [[13.0, 12.0, 11.0], [18.0, 15.0, 16.0]]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(result, expected, atol=1e-06)

def test_calc_square_dist_no_norm():
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]])
    y = torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], [[8.0, 9.0, 10.0], [11.0, 12.0, 13.0]]])
    result = calc_square_dist(x, y, False)
    expected = torch.tensor([[[5.0, 4.0, 3.0], [8.0, 5.0, 6.0]], [[13.0, 12.0, 11.0], [18.0, 15.0, 16.0]]])
    with pytest.raises(RuntimeError):
        assert torch.allclose(result, expected, atol=1e-06)
if __name__ == '__main__':
    pytest.main()