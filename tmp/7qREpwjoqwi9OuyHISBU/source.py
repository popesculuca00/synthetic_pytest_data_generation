import torch

def cross_squared_distance_matrix(x, y, device):
    
    x_norm_squared = torch.sum(torch.mul(x, x), 1)
    y_norm_squared = torch.sum(torch.mul(y, y), 1)

    # Expand so that we can broadcast.
    x_norm_squared_tile = x_norm_squared.unsqueeze(1)
    y_norm_squared_tile = y_norm_squared.unsqueeze(0)

    x_y_transpose = torch.matmul(x.to(device), torch.transpose(y, 0, 1))

    # squared_dists[i,j] = ||x_i - y_j||^2 = x_i'x_i- 2x_i'x_j + x_j'x_j
    squared_dists = x_norm_squared_tile.to(device) - 2 * x_y_transpose + y_norm_squared_tile

    return squared_dists