import pytest
import torch
from source import dice_score_tensor

def test_dice_score_tensor():
    reference = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])
    predictions = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    expected_output = torch.tensor([0.5, 0.5, 0.5])
    assert torch.isclose(dice_score_tensor(reference, predictions), expected_output).all(), 'Dice score tensor test failed'

if __name__ == "__main__":
    pytest.main()