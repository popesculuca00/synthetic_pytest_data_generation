import torch

def transform_homogeneous(matrices, vertices):
    
    if len(matrices.shape) != 3:
        raise ValueError(
            'matrices must have 3 dimensions (missing batch dimension?)')
    if len(vertices.shape) != 3:
        raise ValueError(
            'vertices must have 3 dimensions (missing batch dimension?)')
    homogeneous_coord = torch.ones([vertices.shape[0], vertices.shape[1], 1]).to(vertices.device)
    vertices_homogeneous = torch.cat([vertices, homogeneous_coord], 2)

    return torch.matmul(vertices_homogeneous, matrices.transpose(1, 2))