import torch

def generate_transformation_matrix(camera_position, look_at, camera_up_direction):
    r
    z_axis = (camera_position - look_at)
    z_axis /= z_axis.norm(dim=1, keepdim=True)
    x_axis = torch.cross(camera_up_direction, z_axis, dim=1)
    x_axis /= x_axis.norm(dim=1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    rot_part = torch.stack([x_axis, y_axis, z_axis], dim=2)
    trans_part = (-camera_position.unsqueeze(1) @ rot_part)
    return torch.cat([rot_part, trans_part], dim=1)