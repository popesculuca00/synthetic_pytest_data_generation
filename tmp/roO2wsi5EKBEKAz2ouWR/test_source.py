import pytest
import torch
import numpy as np

# Import the source module
from source import gradient

def test_gradient_autodiff():
    # Define a simple function
    def f(u):
        return u**2
    
    # Analytical gradient
    grad_analytical = torch.tensor([2.0], dtype=torch.float32)
    
    # Finite difference gradient
    u = torch.tensor([1.0], dtype=torch.float32)
    grad_fd = gradient(u, f, method='autodiff')
    
    # Check whether the two gradients are close
    assert torch.allclose(grad_fd, grad_analytical, atol=1e-4), \
        'Autodiff gradient test failed'

def test_gradient_finitediff():
    # Define a two dimensional vector field
    def f(u):
        return u[:, 0]**2 + u[:, 1]**2
    
    # Autodiff and finite difference gradients
    u = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    grad_fd = gradient(u, f, method='finitediff')
    
    # Check whether the two gradients are close
    assert torch.allclose(grad_fd, torch.tensor([[2.0, 4.0], [2.0, 4.0]], dtype=torch.float32), atol=1e-4), \
        'Finitediff gradient test failed'

# Run the tests
if __name__ == "__main__":
    test_gradient_autodiff()
    test_gradient_finitediff()