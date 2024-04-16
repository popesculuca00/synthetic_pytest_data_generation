import torch

def _dice_loss(preds, targets, power=0.2, pos_weight=1., neg_weight=1.):
    

    _, h, w = preds.shape
    _preds = preds.flatten(1)
    _tgts = targets.flatten(1)

    tgt_area = (_tgts).sum(-1)

    pos_mask_det = torch.nonzero((tgt_area != 0), as_tuple=False)
    neg_mask_det = torch.nonzero((tgt_area == 0), as_tuple=False)

    # positive predictions
    _pp = _preds[pos_mask_det].squeeze()
    _tp = _tgts[pos_mask_det].squeeze()

    if _pp.ndim == 1:
        _pp = _pp.unsqueeze(0)
    if _tp.ndim == 1:
        _tp = _tp.unsqueeze(0)

    numerator = (2 * (_pp * _tp).sum(1))
    denominator = (_pp).sum(-1) + (_tp).sum(-1)
    loss_pos = 1 - ((numerator + 1) / (denominator + 1))

    # negative samples (no object) -> invert masks
    _p = (1 - _preds[neg_mask_det].squeeze())
    _t = (1 - _tgts[neg_mask_det].squeeze())

    numerator = 2 * (_p * _t).mean(1)
    denominator = (_p + _t).mean(-1)
    loss_neg = torch.pow(-torch.log(((numerator + 1) / (denominator + 1))) + 1e-4, power)

    return loss_pos.mean() * pos_weight, loss_neg.mean() * neg_weight