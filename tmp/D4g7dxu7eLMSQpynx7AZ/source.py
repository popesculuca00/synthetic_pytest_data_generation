import torch

def Jaccard_loss_cal(true, logits, eps=1e-7):
    
    num_classes = logits.shape[1]
    if true.shape[1] == 2:
        true_1_hot = true.float()
        probas = logits[:,1,:,:].to(1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = logits[:,1,:,:].to(1)
    true_1_hot = true_1_hot.type(logits.type())[:,1,:,:].to(1)
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dim=(1,2))
    cardinality = torch.sum(probas + true_1_hot, dim=(1,2))
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1. - jacc_loss)