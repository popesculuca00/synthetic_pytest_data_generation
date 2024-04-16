import torch

def azimuthal_average(image, center=None):
    # modified to tensor inputs from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    
    # Check input shapes
    assert center is None or (len(center) == 2), f'Center has to be None or len(center)=2 ' \
                                                 f'(but it is len(center)={len(center)}.'
    # Calculate the indices from the image
    H, W = image.shape[-2:]
    h, w = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    if center is None:
        center = torch.tensor([(w.max() - w.min()) / 2.0, (h.max() - h.min()) / 2.0])

    # Compute radius for each pixel wrt center
    r = torch.stack([w-center[0], h-center[1]]).norm(2, 0)

    # Get sorted radii
    r_sorted, ind = r.flatten().sort()
    i_sorted = image.flatten(-2, -1)[..., ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.long()             # attribute to the smaller integer

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented, computes bin change between subsequent radii
    rind = torch.where(deltar)[0]  # location of changed radius

    # compute number of elements in each bin
    nind = rind + 1         # number of elements = idx + 1
    nind = torch.cat([torch.tensor([0]), nind, torch.tensor([H*W])])        # add borders
    nr = nind[1:] - nind[:-1]  # number of radius bin, i.e. counter for bins belonging to each radius

    # Cumulative sum to figure out sums for each radius bin
    if H % 2 == 0:
        raise NotImplementedError('Not sure if implementation correct, please check')
        rind = torch.cat([torch.tensor([0]), rind, torch.tensor([H * W - 1])])  # add borders
    else:
        rind = torch.cat([rind, torch.tensor([H * W - 1])])  # add borders
    csim = i_sorted.cumsum(-1, dtype=torch.float64)             # integrate over all values with smaller radius
    tbin = csim[..., rind[1:]] - csim[..., rind[:-1]]
    # add mean
    tbin = torch.cat([csim[:, 0:1], tbin], 1)

    radial_prof = tbin / nr.to(tbin.device)         # normalize by counted bins

    return radial_prof