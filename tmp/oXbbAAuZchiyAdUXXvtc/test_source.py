import sys
sys.path.append('.')
import source
import torch

def test_accuracy():
    scores = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    targets = torch.tensor([0, 1, 2])
    k = 2
    result = source.accuracy(scores, targets, k)
    assert result == 66.66666666666667, 'The accuracy function is not working as expected'