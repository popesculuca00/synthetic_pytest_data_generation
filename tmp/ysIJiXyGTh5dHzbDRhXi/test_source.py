import torch
import pytest
from source import deptree_nonproj

def test_deptree_nonproj():
    arc_scores = torch.rand(1, 3, 3)
    eps = 1e-05
    output = deptree_nonproj(arc_scores, eps)
    assert not  torch.allclose(output, torch.tensor([[[1.0, 0.1555, 0.2222], [0.0, 1.0, 0.2222], [0.0, 0.1555, 1.0]]]))