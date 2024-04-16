# test_source.py

import pytest
import torch
from source import bboxes_iou

def test_bboxes_iou():
    bboxes_a = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
    bboxes_b = torch.tensor([[5, 5, 15, 15], [5, 5, 10, 10]])

    # Assertion to test whether the function is returning expected output or not
    assert torch.allclose(bboxes_iou(bboxes_a, bboxes_b), torch.tensor([[5.9166666884471401, 0.], 
                                                                      [0., 5.9166666884471401]])), 'The function did not return the expected output'

if __name__ == "__main__":
    test_bboxes_iou()