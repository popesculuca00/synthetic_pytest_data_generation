import pytest
import torch
from source import _pairwise_distances

def test_pairwise_distances():
    # Create random embeddings
    embeddings = torch.randn(10, 10)
    
    # Call the function and get the result
    result = _pairwise_distances(embeddings, squared=False)
    
    # Here we just check if the output shape is correct. 
    # A more complex assertion could be done to check if the result is correct.
    assert result.shape == embeddings.shape