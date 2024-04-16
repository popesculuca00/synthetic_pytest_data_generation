import torch

def rae(target, predictions: list, total=True):
    

    y_hat_test = predictions[0]
    y_hat_naive = torch.mean(target)

    if not total:
        raise NotImplementedError("rae does not support loss over the horizon")

    # denominator is the mean absolute error of the preidicity dependent "naive forecast method"
    # on the test set -->outsample
    return torch.mean(torch.abs(target - y_hat_test)) / torch.mean(torch.abs(target - y_hat_naive))