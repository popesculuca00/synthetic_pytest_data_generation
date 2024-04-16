import torch

def transform_pointcloud(pc, transformation_matrix, in_place=True):
    
    assert isinstance(pc, torch.Tensor)
    assert type(pc) == type(transformation_matrix)
    assert pc.ndim == transformation_matrix.ndim
    if pc.ndim == 3:
        N, M = 1, 2
    elif pc.ndim == 2:
        N, M = 0, 1
    else:
        raise Exception("Pointcloud must have dimension Nx3 or BxNx3")
    xyz = pc[..., :3]
    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    homogeneous_xyz = torch.cat((xyz, torch.ones(ones_dim, device=xyz.device)), dim=M)
    transformed_xyz = torch.matmul(
        transformation_matrix, homogeneous_xyz.transpose(N, M)
    )
    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
        return pc
    return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)