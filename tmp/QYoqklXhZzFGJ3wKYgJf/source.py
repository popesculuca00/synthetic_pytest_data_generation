import torch

def linear(input, weight, bias=None, scale=None, zero_point=None):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[float], Optional[int]) -> Tensor
    r
    if scale is None:
        scale = input.q_scale()
    if zero_point is None:
        zero_point = input.q_zero_point()
    _packed_params = torch.ops.quantized.linear_prepack(weight, bias)
    return torch.ops.quantized.linear(input, _packed_params, scale, zero_point)