import torch

def mrr(pred, target, k):
    r
    row, col = torch.nonzero(pred[:, :k], as_tuple=True)
    row_uniq, counts = torch.unique_consecutive(row, return_counts=True)
    idx = torch.zeros_like(counts)
    idx[1:] = counts.cumsum(dim=-1)[:-1]
    first = col.new_zeros(pred.size(0)).scatter_(0, row_uniq, col[idx]+1)
    output = 1.0 / first
    output[first == 0] = 0
    return output.mean()