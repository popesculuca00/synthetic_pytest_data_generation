import torch

def sharpness(predictions:list, total = True):
    

    assert len(predictions) == 2
    y_pred_upper = predictions[0]
    y_pred_lower = predictions[1]
    if total:
        return torch.mean(y_pred_upper - y_pred_lower)
    else:
        return torch.mean(y_pred_upper - y_pred_lower, dim=0)