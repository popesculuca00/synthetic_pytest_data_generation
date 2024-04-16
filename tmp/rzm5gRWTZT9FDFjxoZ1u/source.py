import torch

def check_bounds(myt, imsize):
    
    xt = myt[:,0]
    yt = myt[:,1]
    x_out = (torch.floor(xt) < 0) | (torch.ceil(xt) >= imsize[0])
    y_out = (torch.floor(yt) < 0) | (torch.ceil(yt) >= imsize[1])
    out = x_out | y_out

    return out