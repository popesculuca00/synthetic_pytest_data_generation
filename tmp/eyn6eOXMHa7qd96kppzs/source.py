def pixel_wise_boundary_precision_recall(pred, gt):
    
    tp = float((gt * pred).sum())
    fp = (pred * (1-gt)).sum()
    fn = (gt * (1-pred)).sum()
    return tp/(tp+fp), tp/(tp+fn)