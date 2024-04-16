import pytest
import torch
from source import construct_edge_feature_gather

def test_construct_edge_feature_gather():
    feature = torch.randn(2, 3, 10)
    knn_inds = torch.randint(0, 10, (2, 3, 10))
    edge_feature = construct_edge_feature_gather(feature, knn_inds)

    assert edge_feature.shape == (2, 6, 10, 10)