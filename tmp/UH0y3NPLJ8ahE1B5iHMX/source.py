import torch

def invert_pose(T01):
    
    Tinv = torch.eye(4, device=T01.device, dtype=T01.dtype).repeat([len(T01), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T01[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T01[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv