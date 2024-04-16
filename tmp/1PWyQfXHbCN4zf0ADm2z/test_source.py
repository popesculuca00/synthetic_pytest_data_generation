import torch
import pytest

from source import get_accuracy

def test_get_accuracy():
    # Given
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    targets = torch.tensor([0, 1, 2])
    
    # When
    result = get_accuracy(logits, targets)
    
    # Then
    assert result == 1.0, "The accuracy should be 1.0 when all predictions are correct"

def test_get_accuracy_with_incorrect_predictions():
    # Given
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    targets = torch.tensor([0, 2, 1])
    
    # When
    result = get_accuracy(logits, targets)
    
    # Then
    assert result != 1.0, "The accuracy should not be 1.0 when there are incorrect predictions"