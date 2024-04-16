import torch

def _pairwise_union_regions(boxes1, boxes2):
    
    
    X1 = torch.min(boxes1[:, None, 0], boxes2[:, 0]).flatten()
    Y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1]).flatten()
    X2 = torch.max(boxes1[:, None, 2], boxes2[:, 2]).flatten()
    Y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3]).flatten()

    unions = torch.stack([X1, Y1, X2, Y2], dim=1)
    # unions = Boxes(unions) # BoxMode.XYXY_ABS

    return unions