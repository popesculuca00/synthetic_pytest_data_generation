import torch

def cheb_conv(x, weights, laplacian=None):
    

    B, Cin, V = x.shape  # (B, Cin, V)
    R, _, _ = weights.shape  # (R, Cin, Cout)

    if laplacian is None and R > 1:
        raise ValueError(f"Can't perform Chebyschev convolution without laplacian if R > 1")

    x0 = x.permute(2, 0, 1).contiguous().view(V, B * Cin)  # (B, Cin, V) -> (V, B*Cin)
    x = x0.unsqueeze(0)  # (V, B*Cin) -> (1, V, B*Cin)

    if R > 1:
        x1 = torch.mm(laplacian, x0)  # (V, B*Cin)
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # (1, V, B*Cin) -> (2, V, B*Cin)

        for _ in range(2, R):
            x2 = 2 * torch.mm(laplacian, x1) - x0  # -> (V, B*Cin)
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # (k-1, V, B*Cin) -> (k, V, B*Cin)
            x0, x1 = x1, x2  # (V, B*Cin), (V, B*Cin)

    x = x.contiguous().view(R, V, B, Cin)  # (R, V, B*Cin) -> (R, V, B, Cin)
    x = torch.tensordot(x, weights, dims=([0, 3], [0, 1]))  # (V, B, Cout)

    return x