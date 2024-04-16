import torch

def nll_gauss(target, predictions:list, total = True):
    

    assert len(predictions) == 2
    expected_value = predictions[0]
    log_variance = predictions[1]

    # y, y_pred, var_pred must have the same shape
    assert target.shape == expected_value.shape # target.shape = torch.Size([batchsize, horizon, # of target variables]) e.g.[64,40,1]
    assert target.shape == log_variance.shape

    squared_errors = (target - expected_value) ** 2
    if total:
        return torch.mean(squared_errors / (2 * log_variance.exp()) + 0.5 * log_variance)
    else:
        return torch.mean(squared_errors / (2 * log_variance.exp()) + 0.5 * log_variance, dim=0)