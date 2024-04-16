import torch

def mape(target, predictions:list, total = True):
    
    
    if not total:
        raise NotImplementedError("MAPE does not support loss over the horizon")

    return torch.mean(torch.abs((target - predictions[0]) / target)) * 100