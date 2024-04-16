import pytest
import torch
from source import compas_robustness_loss

def test_compas_robustness_loss():
    x = torch.randn(10, 10)
    aggregates = torch.randn(10, 3)
    concepts = torch.randn(10, 3)
    relevances = torch.randn(10, 3)

    output = compas_robustness_loss(x, aggregates, concepts, relevances)
    assert torch.allclose(output, torch.zeros_like(output)), "The output is not as expected"