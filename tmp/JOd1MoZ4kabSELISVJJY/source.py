import torch

def one_hot_encode(tensor, num_classes: int):
    
    tensor = tensor.long()
    if tensor.dim() == 0:
        return torch.scatter(torch.zeros(num_classes), -1, tensor, 1)
    else:
        tensor_ = tensor.reshape(-1, 1)
        out = torch.scatter(torch.zeros(tensor_.shape[0], num_classes), -1, tensor_, 1)
        return out.reshape(tensor.shape + (num_classes,))