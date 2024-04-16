import torch
import pytest
from source import Jaccard_loss_cal  # assuming that the function is in source.py

def test_jaccard_loss_cal():
    true = torch.randint(0, 2, (10,))
    logits = torch.rand((10, 2, 10, 10))
    result = Jaccard_loss_cal(true, logits)
    assert torch.isclose(result, 0.0, atol=1e-4), "Test Failed!"

def test_jaccard_loss_cal_with_2_classes():
    true = torch.randint(0, 2, (10,))
    true[true == 1] = 0
    logits = torch.rand((10, 2, 10, 10))
    logits[:, 0, :, :] = 0
    result = Jaccard_loss_cal(true, logits)
    assert torch.isclose(result, 0.5, atol=1e-4), "Test Failed!"

if __name__ == "__main__":
    test_jaccard_loss_cal()
    test_jaccard_loss_cal_with_2_classes()