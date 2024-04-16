import torch

def pairwise_orthogonalization_torch(v1, v2, center:bool=False):
    

    assert v1.ndim == v2.ndim
    if v1.ndim==1:
        v1 = v1[:,None]
        v2 = v2[:,None]
    assert v1.shape[1] == v2.shape[1]
    assert v1.shape[0] == v2.shape[0]
    
    if center:
        v1 = v1 - torch.mean(v1, dim=0)
        v2 = v2 - torch.mean(v2, dim=0)

    # v1_orth = v1 - (torch.diag(torch.matmul(v1.T, v2)) / torch.diag(torch.matmul(v2.T, v2)))*v2
    v1_orth = v1 - (torch.sum(v1 * v2, dim=0) / torch.sum(v2 * v2, dim=0) )*v2

    v1_var = torch.var(v1, dim=0)
    EVR = 1 - (torch.var(v1_orth, dim=0) / v1_var)

    EVR_total_weighted = torch.sum(v1_var * EVR) / torch.sum(v1_var)
    EVR_total_unweighted = torch.mean(EVR)
    return v1_orth.squeeze(), EVR, EVR_total_weighted, EVR_total_unweighted