import torch

def get_accuracy(logits, targets):
    
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())