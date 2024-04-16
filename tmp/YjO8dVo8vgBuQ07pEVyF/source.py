import torch

def heatmap(xdelta:float, ydelta:float, scale:float=10., precision:float=15., heatmap_dim:int=19):
    
    grid = torch.linspace(-1, 1, heatmap_dim)
    g1, g0 = torch.meshgrid(grid, grid)
    out = scale * torch.exp(-1 * precision * ((g0 - xdelta)**2 + (g1 - ydelta)**2))
    return out