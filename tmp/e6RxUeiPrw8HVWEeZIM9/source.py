import torch

def extract_odms(voxelgrids):
    r
    # Cast input to torch.bool to make it run faster.
    voxelgrids = voxelgrids.bool()
    device = voxelgrids.device
    dtype = voxelgrids.dtype

    dim = voxelgrids.shape[-1]
    batch_num = voxelgrids.shape[0]

    multiplier = torch.arange(1, dim + 1, device=device)
    reverse_multiplier = torch.arange(dim, 0, step=-1, device=device)
    full_multiplier = torch.cat([multiplier, reverse_multiplier], dim=0)

    # z_axis
    z_axis = voxelgrids.unsqueeze(1) * full_multiplier.view(1, 2, 1, 1, -1)
    z_axis_values, _ = torch.max(z_axis, dim=4)

    # y_axis
    y_axis = voxelgrids.unsqueeze(1) * full_multiplier.view(1, 2, 1, -1, 1)
    y_axis_values, _ = torch.max(y_axis, dim=3)

    # x_axis
    x_axis = voxelgrids.unsqueeze(1) * full_multiplier.view(1, 2, -1, 1, 1)
    x_axis_values, _ = torch.max(x_axis, dim=2)
    return dim - torch.cat([z_axis_values, y_axis_values, x_axis_values], dim=1)