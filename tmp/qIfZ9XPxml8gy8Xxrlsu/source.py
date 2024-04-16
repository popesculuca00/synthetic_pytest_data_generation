import torch

def eval_acc(pred, labels, mask=None):
    r

    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0

    acc = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

    return acc