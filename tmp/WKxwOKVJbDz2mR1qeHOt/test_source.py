import torch
import numpy as np
import source  # assuming the source code is in a file named source.py

def test_mrr():
    pred = torch.tensor([[0, 1, 1, 0, 0], [1, 1, 1, 0, 1], [0, 0, 0, 1, 1]])
    target = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 0]])
    k = 3
    assert np.isclose(source.mrr(pred, target, k), 0.75, atol=1e-6)

test_mrr()