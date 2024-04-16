# import the necessary package
import pytest
from source import captum_sequence_forward
from transformers import AutoModel
import torch

def test_captum_sequence_forward():
    # initialize model
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    # sample input
    inputs = torch.tensor([[0,1,2,3,4,5]])
    attention_mask = torch.tensor([[1,1,1,1,1,0]])

    # single assertion per test
    assert captum_sequence_forward(inputs, attention_mask=attention_mask, model=model) == 0