def compute_AUC(x, y, reorder=False):
  
  from sklearn.metrics import auc
  return auc(x, y, reorder)