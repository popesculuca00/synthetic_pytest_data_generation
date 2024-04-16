import torch

def categorical_focal_loss(y_true, y_pred):
    
    gamma = 2.
    alpha = .25

    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= torch.sum(y_pred, dim=-1, keepdim=True)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = 1e-07
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * torch.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * torch.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return torch.mean(torch.mean(loss, dim=1), dim=0)