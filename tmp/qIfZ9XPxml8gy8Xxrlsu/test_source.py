from source import *
import pytest
import torch
import sys
sys.path.append('.')
from source import eval_acc

def test_eval_acc_with_mask():
    pred = torch.tensor([[0.1, 0.9, 0.7], [0.3, 0.1, 0.2]])
    labels = torch.tensor([1, 2, 0])
    mask = torch.tensor([True, False, True])
    assert eval_acc(pred, labels, mask) == 0.5

def test_eval_acc_without_mask():
    pred = torch.tensor([[0.1, 0.9, 0.7], [0.3, 0.1, 0.2]])
    labels = torch.tensor([1, 2, 0])
    assert eval_acc(pred, labels) == 0.5

def test_eval_acc_with_none():
    pred = None
    labels = torch.tensor([1, 2, 0])
    mask = torch.tensor([True, True, True])
    assert eval_acc(pred, labels, mask) == 0.0

def test_eval_acc_empty_tensor():
    pred = torch.tensor([])
    labels = torch.tensor([])
    mask = torch.tensor([])
    assert eval_acc(pred, labels, mask) == 0.0