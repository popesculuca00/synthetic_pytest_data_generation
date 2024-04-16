import pytest
import torch
from source import gradient_to_excitation_backprop_saliency

def test_gradient_to_excitation_backprop_saliency():
    # Given
    input_tensor = torch.rand(1, requires_grad=True)
    output = gradient_to_excitation_backprop_saliency(input_tensor)

    # When
    grad = torch.rand(1)
    input_tensor.backward(grad)

    # Then
    assert torch.allclose(output, input_tensor.grad)