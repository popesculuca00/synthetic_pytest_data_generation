import torch

def euler_matrices(angles):
    
    s = torch.sin(angles)
    c = torch.cos(angles)
    # Rename variables for readability in the matrix definition below.
    c0, c1, c2 = (c[:, 0], c[:, 1], c[:, 2])
    s0, s1, s2 = (s[:, 0], s[:, 1], s[:, 2])

    zeros = torch.zeros_like(s[:, 0])
    ones = torch.ones_like(s[:, 0])

    # pyformat: disable
    flattened = torch.cat(
        [
            c2 * c1, c2 * s1 * s0 - c0 * s2, s2 * s0 + c2 * c0 * s1, zeros,
            c1 * s2, c2 * c0 + s2 * s1 * s0, c0 * s2 * s1 - c2 * s0, zeros,
            -s1, c1 * s0, c1 * c0, zeros,
            zeros, zeros, zeros, ones
        ],
        dim=0)
    # pyformat: enable
    reshaped = flattened.view(4, 4, -1)
    return reshaped.transpose(2, 0).transpose(2, 1)  # TODO