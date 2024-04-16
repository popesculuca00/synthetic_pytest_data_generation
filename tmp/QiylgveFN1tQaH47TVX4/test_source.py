import sys
sys.path.append(".") 

import pytest
import torch
from source import one_hot

def test_one_hot():
    indices = torch.tensor([0,1,2,3])
    depth = 5
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    indices = indices.to(device)
    expected_output = torch.zeros(indices.size() + torch.Size([depth]))
    expected_output = expected_output.to(device)
    expected_output.scatter_(1, indices.view(indices.size()+torch.Size([1])), 1)
    
    output = one_hot(indices, depth)
    
    assert torch.allclose(output, expected_output)