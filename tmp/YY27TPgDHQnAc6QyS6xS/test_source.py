import pytest
from source import captum_sequence_forward
from transformers import AutoModelForSequenceClassification
import torch

def test_captum_sequence_forward():
    # Assuming the model is a pre-trained BERT model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    inputs = torch.tensor([[0, 1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

    # Test when position is 0
    pred = captum_sequence_forward(inputs, attention_mask=attention_mask, position=0, model=model)
    assert pred.shape == torch.Size([768])  # BERT-base has 768 output dimensions

    # Test when position is not 0
    pred = captum_sequence_forward(inputs, attention_mask=attention_mask, position=1, model=model)
    assert pred.shape == torch.Size([768])
    
    # Test when model is None
    pred = captum_sequence_forward(inputs, attention_mask=attention_mask, position=0, model=None)
    assert pred is None