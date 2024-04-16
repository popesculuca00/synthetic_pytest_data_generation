import torch

def so3_rotation_angle(R, eps: float = 1e-4, cos_angle: bool = False):
    

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # clamp to valid range
    rot_trace = torch.clamp(rot_trace, -1.0, 3.0)

    # phi ... rotation angle
    phi = 0.5 * (rot_trace - 1.0)

    if cos_angle:
        return phi
    else:
        # pyre-fixme[16]: `float` has no attribute `acos`.
        return phi.acos()