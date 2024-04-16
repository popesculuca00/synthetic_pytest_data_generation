import torch

def square_distance(xyz1, xyz2):
    
    # base: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
    inner = -2*torch.matmul(xyz1.transpose(2, 1), xyz2)
    xyz_column = torch.sum(xyz2**2, dim=1, keepdim=True)
    xyz_row = torch.sum(xyz1**2, dim=1, keepdim=True).transpose(2, 1)
    square_dist = torch.sqrt(xyz_column + inner + xyz_row)
    return square_dist