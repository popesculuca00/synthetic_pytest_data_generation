import torch

def dice_score_tensor(reference, predictions):
    
    eps = 1.
    ab = torch.sum(reference * predictions, dim=(1, 2, 3))
    a = torch.sum(reference, dim=(1, 2, 3))
    b = torch.sum(predictions, dim=(1, 2, 3))
    dsc = (2 * ab + eps) / (a + b + eps)
    dsc = torch.mean(dsc)
    return dsc