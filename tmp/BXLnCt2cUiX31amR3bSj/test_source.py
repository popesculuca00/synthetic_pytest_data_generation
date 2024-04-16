import pytest
import torch
from .source import compute_ece as src_compute_ece

def test_compute_ece():
    prob = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
    bin_mean_prob = torch.tensor([0.4, 0.6])

    result = src_compute_ece(prob, bin_mean_prob)

    assert torch.isclose(result, torch.tensor(0.2)), "The computed ECE value is not correct"