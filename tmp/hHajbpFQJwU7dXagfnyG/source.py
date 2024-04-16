import torch

def weighted_sigmoid_log_loss(positive_predictions, negative_predictions, candidate_predictions, weight, alpha=1.0):
    
    loss1 = -torch.log(torch.sigmoid(positive_predictions))
    loss0 = -torch.log(1 - torch.sigmoid(negative_predictions))

    loss_cand = -torch.log(torch.sigmoid(candidate_predictions))

    if weight is not None:
        loss_cand = loss_cand * weight.expand_as(loss_cand)

    if alpha is not None:
        loss_cand = loss_cand * alpha

    loss = torch.sum(torch.cat((loss1, loss0, loss_cand), 1), dim=1)
    reg_loss = torch.sum(torch.cat((loss1, loss0), 1), dim=1)

    return loss.mean(), reg_loss.mean()