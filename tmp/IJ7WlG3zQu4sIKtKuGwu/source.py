import torch

def draw_shape(pos, sigma_x, sigma_y, angle, size):
    
    device = pos.device
    assert sigma_x.device == sigma_y.device == angle.device == device, "inputs should be on the same device!"

    # create 2d meshgrid
    x, y = torch.meshgrid(torch.arange(0, size), torch.arange(0, size))
    x, y = x.unsqueeze(0).unsqueeze(0).to(device), y.unsqueeze(0).unsqueeze(0).to(device)

    # see https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    a = torch.cos(angle) ** 2 / (2 * sigma_x ** 2) + torch.sin(angle) ** 2 / (2 * sigma_y ** 2)
    b = -torch.sin(2 * angle) / (4 * sigma_x ** 2) + torch.sin(2 * angle) / (4 * sigma_y ** 2)
    c = torch.sin(angle) ** 2 / (2 * sigma_x ** 2) + torch.cos(angle) ** 2 / (2 * sigma_y ** 2)

    # append dimsensions for broadcasting
    pos = pos.view(1, 1, 2, 1, 1)
    a, b, c = a.view(1, 1), b.view(1, 1), c.view(1, 1)

    # pixel-wise distance from center
    xdist = (x - pos[:, :, 0])
    ydist = (y - pos[:, :, 1])

    # gaussian function
    g = torch.exp((-a * xdist ** 2 - 2 * b * xdist * ydist - c * ydist ** 2))

    return g