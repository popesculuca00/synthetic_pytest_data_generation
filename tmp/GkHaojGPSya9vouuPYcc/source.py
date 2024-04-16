import torch

def is_complex_symmetric(z: torch.Tensor, atol=3e-5, rtol=1e-5):
    
    real_z, imag_z = z.real, z.imag
    return torch.allclose(
        real_z, real_z.transpose(-1, -2), atol=atol, rtol=rtol
    ) and torch.allclose(imag_z, imag_z.transpose(-1, -2), atol=atol, rtol=rtol)